#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

// max number of distinct op keys tracked by GGML_METAL_PROFILE=1
#define GGML_METAL_PROF_MAX 512

struct ggml_metal_prof_entry {
    char    key[112];
    int64_t n;
    double  t;     // GPU time, seconds
    double  bytes; // src0 bytes for mat-mul ops
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;

    // GGML_METAL_PROFILE: 1 - GPU time of each op, 2 - GPU time of whole graphs
    int profile;
    id<MTLCounterSampleBuffer> prof_sb;
    struct ggml_metal_prof_entry * prof;
    int     prof_n;
    int64_t prof_n_graphs;
    double  prof_t_gpu;  // sum of op times (1) or of graph GPU spans (2)
    double  prof_t_busy; // mode 2: sum of command buffer GPU times
    double  prof_t_enc;  // CPU time spent in graph_compute
    double  prof_t_gap;  // mode 1: GPU time between encoders
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    {
        const char * val = getenv("GGML_METAL_PROFILE");
        res->profile = val ? atoi(val) : 0;
        if (res->profile > 0) {
            res->prof = calloc(GGML_METAL_PROF_MAX, sizeof(struct ggml_metal_prof_entry));
            GGML_LOG_INFO("%s: profile mode       = %d\n", __func__, res->profile);
        }
    }

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

static void ggml_metal_prof_graph_span(ggml_metal_t ctx);
static void ggml_metal_prof_print(ggml_metal_t ctx);

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ctx->profile > 0) {
        if (ctx->profile == 2) {
            ggml_metal_prof_graph_span(ctx);
        }
        ggml_metal_prof_print(ctx);
        free(ctx->prof);
        if (ctx->prof_sb) {
            [ctx->prof_sb release];
        }
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

static void ggml_metal_prof_add(ggml_metal_t ctx, const char * key, double t, double bytes) {
    for (int i = 0; i < ctx->prof_n; ++i) {
        struct ggml_metal_prof_entry * e = &ctx->prof[i];
        if (strcmp(e->key, key) == 0) {
            e->n     += 1;
            e->t     += t;
            e->bytes += bytes;
            return;
        }
    }

    if (ctx->prof_n == GGML_METAL_PROF_MAX) {
        return;
    }

    struct ggml_metal_prof_entry * e = &ctx->prof[ctx->prof_n++];
    snprintf(e->key, sizeof(e->key), "%s", key);
    e->n     = 1;
    e->t     = t;
    e->bytes = bytes;
}

// op + src0 type + node name without the layer suffix
static void ggml_metal_prof_key(char * buf, size_t size, const struct ggml_tensor * node, int n_fuse) {
    char name[64];
    snprintf(name, sizeof(name), "%s", node->name);
    for (char * p = name; *p; ++p) {
        if ((*p == '-' || *p == '_') && p[1] >= '0' && p[1] <= '9') {
            *p = 0;
            break;
        }
    }

    const char * t0 = node->src[0] ? ggml_type_name(node->src[0]->type) : "-";

    char fuse[16] = "";
    if (n_fuse > 1) {
        snprintf(fuse, sizeof(fuse), " +%d", n_fuse - 1);
    }

    snprintf(buf, size, "%-14s %-6s %-26s%s", ggml_op_desc(node), t0, name, fuse);
}

static int ggml_metal_prof_cmp(const void * a, const void * b) {
    const double ta = ((const struct ggml_metal_prof_entry *) a)->t;
    const double tb = ((const struct ggml_metal_prof_entry *) b)->t;
    return ta < tb ? 1 : (ta > tb ? -1 : 0);
}

static void ggml_metal_prof_graph_span(ggml_metal_t ctx) {
    double t0 = 0.0;
    double t1 = 0.0;
    double busy = 0.0;
    bool   any = false;

    for (int i = 0; i <= ctx->n_cb; ++i) {
        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
        if (!cmd_buf || [cmd_buf status] == MTLCommandBufferStatusNotEnqueued) {
            continue;
        }

        [cmd_buf waitUntilCompleted];

        const double s = [cmd_buf GPUStartTime];
        const double e = [cmd_buf GPUEndTime];

        t0 = any ? MIN(t0, s) : s;
        t1 = any ? MAX(t1, e) : e;
        busy += e - s;
        any = true;
    }

    if (any) {
        ctx->prof_n_graphs += 1;
        ctx->prof_t_gpu    += t1 - t0;
        ctx->prof_t_busy   += busy;

        if (getenv("GGML_METAL_PROFILE_EACH")) {
            fprintf(stderr, "ggml_metal_profile: graph %lld, %d nodes, GPU %.3f ms\n", (long long) ctx->prof_n_graphs, ctx->gf ? ctx->gf->n_nodes : 0, 1e3*(t1 - t0));
        }
    }
}

static void ggml_metal_prof_print(ggml_metal_t ctx) {
    const int64_t ng = MAX(1, ctx->prof_n_graphs);

    fprintf(stderr, "\nggml_metal_profile: mode %d, %lld graphs, GPU %.3f ms/graph, CPU in graph_compute %.3f ms/graph\n",
            ctx->profile, (long long) ctx->prof_n_graphs, 1e3*ctx->prof_t_gpu/ng, 1e3*ctx->prof_t_enc/ng);

    if (ctx->profile == 2) {
        fprintf(stderr, "ggml_metal_profile: sum of command buffer GPU time %.3f ms/graph\n", 1e3*ctx->prof_t_busy/ng);
        return;
    }

    fprintf(stderr, "ggml_metal_profile: GPU time between encoders %.3f ms/graph (not in the table)\n", 1e3*ctx->prof_t_gap/ng);

    qsort(ctx->prof, ctx->prof_n, sizeof(struct ggml_metal_prof_entry), ggml_metal_prof_cmp);

    fprintf(stderr, "%-50s %8s %10s %9s %6s %8s\n", "op / src0 / name", "n/graph", "ms/graph", "us/op", "%", "GB/s");
    for (int i = 0; i < ctx->prof_n; ++i) {
        const struct ggml_metal_prof_entry * e = &ctx->prof[i];
        fprintf(stderr, "%-50s %8.1f %10.3f %9.1f %6.2f %8.1f\n", e->key,
                (double) e->n/ng, 1e3*e->t/ng, 1e6*e->t/e->n, 100.0*e->t/ctx->prof_t_gpu,
                e->bytes > 0 ? e->bytes/e->t/1e9 : 0.0);
    }
    fprintf(stderr, "\n");
}

// number of ops timed per command buffer
#define GGML_METAL_PROF_OPS 1024

static bool ggml_metal_prof_init_sample_buffer(ggml_metal_t ctx) {
    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    id<MTLCounterSet> cs_ts = nil;
    for (id<MTLCounterSet> cs in device.counterSets) {
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            cs_ts = cs;
        }
    }
    if (!cs_ts || ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        GGML_LOG_ERROR("%s: GPU timestamps at encoder boundaries are not supported\n", __func__);
        return false;
    }

    MTLCounterSampleBufferDescriptor * desc = [[MTLCounterSampleBufferDescriptor alloc] init];
    desc.counterSet  = cs_ts;
    desc.storageMode = MTLStorageModeShared;
    desc.sampleCount = 2*GGML_METAL_PROF_OPS;

    NSError * error = nil;
    ctx->prof_sb = [device newCounterSampleBufferWithDescriptor:desc error:&error];
    [desc release];
    if (!ctx->prof_sb) {
        GGML_LOG_ERROR("%s: failed to create counter sample buffer: %s\n", __func__, [[error localizedDescription] UTF8String]);
        return false;
    }

    return true;
}

// GGML_METAL_PROFILE=1: encode each op in its own serial encoder and read the GPU timestamps at the encoder boundaries
// the ops run back to back on the GPU, so the clocks stay as in a normal run, but ops do not overlap
static enum ggml_status ggml_metal_graph_compute_profile(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    @autoreleasepool {
        id<MTLDevice>       device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLCommandQueue> queue  = ggml_metal_device_get_queue(ctx->dev);

        if (!ctx->prof_sb && !ggml_metal_prof_init_sample_buffer(ctx)) {
            return GGML_STATUS_FAILED;
        }

        if (ctx->cmd_buf_last) {
            [ctx->cmd_buf_last waitUntilCompleted];
            ctx->cmd_buf_last = nil;
        }

        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_op_t ctx_op = ggml_metal_op_init(ctx->dev, cmd_buf, gf, 0, gf->n_nodes,
                ctx->use_fusion, false, false, ctx->debug_graph, ctx->debug_fusion);

        const int n_nodes = ggml_metal_op_n_nodes(ctx_op);

        const struct ggml_tensor * nodes[GGML_METAL_PROF_OPS];
        int fused[GGML_METAL_PROF_OPS];

        double t_graph = 0.0;

        int idx = 0;
        while (idx < n_nodes) {
            int n = 0;
            while (idx < n_nodes && n < GGML_METAL_PROF_OPS) {
                const int res = ggml_metal_op_encode_single(ctx_op, cmd_buf, idx, ctx->prof_sb, 2*n, 2*n + 1);
                nodes[n] = ggml_metal_op_node(ctx_op, idx);
                fused[n] = res;
                n++;
                idx += MAX(res, 1);
            }

            MTLTimestamp cpu0, gpu0, cpu1, gpu1;
            [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];

            if ([cmd_buf status] != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: command buffer failed with status %d\n", __func__, (int) [cmd_buf status]);
                ggml_metal_op_free(ctx_op);
                return GGML_STATUS_FAILED;
            }

            // seconds per GPU timestamp tick, the CPU timestamps are in ns
            const double scale = gpu1 > gpu0 ? 1e-9*(double)(cpu1 - cpu0)/(double)(gpu1 - gpu0) : 1e-9;

            NSData * data = [ctx->prof_sb resolveCounterRange:NSMakeRange(0, 2*n)];
            const MTLCounterResultTimestamp * ts = (const MTLCounterResultTimestamp *) [data bytes];

            for (int i = 0; i < n; ++i) {
                const uint64_t t0 = ts[2*i + 0].timestamp;
                const uint64_t t1 = ts[2*i + 1].timestamp;
                if (t0 == MTLCounterErrorValue || t1 == MTLCounterErrorValue || t1 < t0) {
                    continue;
                }

                if (i > 0 && ts[2*i - 1].timestamp != MTLCounterErrorValue && t0 > ts[2*i - 1].timestamp) {
                    ctx->prof_t_gap += scale*(double)(t0 - ts[2*i - 1].timestamp);
                }

                const struct ggml_tensor * node = nodes[i];
                const double t = scale*(double)(t1 - t0);

                double bytes = 0.0;
                if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0]) {
                    bytes = ggml_nbytes(node->src[0]);
                }

                char key[112];
                ggml_metal_prof_key(key, sizeof(key), node, fused[i]);
                ggml_metal_prof_add(ctx, key, t, bytes);

                t_graph += t;
            }

            cmd_buf = [queue commandBuffer];
        }

        ggml_metal_op_free(ctx_op);

        ctx->prof_n_graphs += 1;
        ctx->prof_t_gpu    += t_graph;
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t t_start_us = ctx->profile > 0 ? ggml_time_us() : 0;

    if (ctx->profile == 1) {
        const enum ggml_status status = ggml_metal_graph_compute_profile(ctx, gf);
        ctx->prof_t_enc += 1e-6*(ggml_time_us() - t_start_us);
        return status;
    }

    if (ctx->profile == 2 && ctx->cmd_bufs[ctx->n_cb].obj) {
        ggml_metal_prof_graph_span(ctx);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    if (ctx->profile == 2) {
        ctx->prof_t_enc += 1e-6*(ggml_time_us() - t_start_us);
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
