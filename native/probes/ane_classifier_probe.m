// ane_classifier_probe.m — Stage 2: ANE Classifier as 1×1 Conv
// Our classifier is CPU cblas (768→8192), taking 111ms = 35% of step time.
// Vipul proved ANE classifier as 32K-channel conv is 10.2× faster than CPU cblas.
// This probe: express classifier as 1×1 conv on ANE, benchmark vs CPU cblas.
// Source: Vipul/ANEgpt (PR #19), confirmed by Anemll
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Generate MIL for classifier: 1×1 conv, input [1, DIM, 1, SEQ], weight [VOCAB, DIM, 1, 1]
// Output: [1, VOCAB, 1, SEQ]
static NSString *genClassifierMIL(int dim, int vocab, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:
        @"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        @"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n", vocab, dim, vocab, dim];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n", vocab, seq];
    [m appendString:@"        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", vocab, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSData *buildClassifierWeightBlob(int dim, int vocab) {
    // Weight shape: [VOCAB, DIM, 1, 1] in FP16
    NSUInteger wsize = (NSUInteger)vocab * dim * 2;
    NSUInteger total = 128 + wsize; // 64-byte global header + 64-byte chunk header + data
    uint8_t *buf = calloc(total, 1);
    // Global header
    buf[0] = 1; buf[4] = 2;
    // Chunk header with DEADBEEF magic
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf+72) = (uint32_t)wsize;
    *(uint32_t*)(buf+80) = 128;
    // Random FP16 weights (small values to avoid overflow)
    uint16_t *fp16 = (uint16_t*)(buf + 128);
    for (NSUInteger j = 0; j < (NSUInteger)vocab * dim; j++)
        fp16[j] = (arc4random() & 0x03FF) | 0x2000; // small positive FP16
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("=== Stage 2: ANE Classifier Probe ===\n");
        printf("Source: Vipul/ANEgpt, confirmed by Anemll\n");
        printf("Express classifier (DIM→VOCAB matmul) as 1×1 conv on ANE\n\n");

        // Our actual dimensions
        int dim = 768;
        int vocab = 8192;
        int seq = 512; // also test with 1024

        int seqs[] = {512, 1024};
        for (int si = 0; si < 2; si++) {
            seq = seqs[si];
            printf("========================================\n");
            printf("Testing: DIM=%d, VOCAB=%d, SEQ=%d\n", dim, vocab, seq);
            printf("Weight size: %.1f MB\n", (double)dim * vocab * 2 / 1024 / 1024);
            printf("========================================\n\n");

            // =========================================
            // TEST 1: CPU cblas classifier (current approach)
            // =========================================
            printf("--- CPU cblas (current) ---\n");
            {
                // Allocate FP32 buffers
                float *x_f32 = malloc(dim * seq * sizeof(float));
                float *w_f32 = malloc(dim * vocab * sizeof(float));
                float *y_f32 = malloc(vocab * seq * sizeof(float));

                // Fill with random data
                for (int i = 0; i < dim * seq; i++) x_f32[i] = (float)(arc4random() & 0xFFFF) / 65536.0f - 0.5f;
                for (int i = 0; i < dim * vocab; i++) w_f32[i] = (float)(arc4random() & 0xFFFF) / 65536.0f * 0.01f;

                // Warmup
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq, vocab, dim, 1.0f, x_f32, dim, w_f32, dim, 0.0f, y_f32, vocab);

                int iters = 20;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq, vocab, dim, 1.0f, x_f32, dim, w_f32, dim, 0.0f, y_f32, vocab);
                }
                double cpu_ms = ticksToMs(mach_absolute_time() - t0) / iters;
                double gflops = 2.0 * seq * vocab * dim / 1e9;
                printf("  %d iters, avg: %.2f ms/eval\n", iters, cpu_ms);
                printf("  GFLOPS: %.1f\n", gflops / (cpu_ms / 1000));
                printf("  First output: %.4f\n\n", y_f32[0]);

                free(x_f32); free(w_f32); free(y_f32);
            }

            // =========================================
            // TEST 2: ANE classifier (1×1 conv)
            // =========================================
            printf("--- ANE 1×1 conv classifier ---\n");
            {
                NSError *e = nil;
                NSData *milData = [[genClassifierMIL(dim, vocab, seq) dataUsingEncoding:NSUTF8StringEncoding] copy];
                NSData *wb = buildClassifierWeightBlob(dim, vocab);

                Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
                Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
                Class AR   = NSClassFromString(@"_ANERequest");
                Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

                NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}};
                id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
                if (!desc) { printf("  FAIL: descriptor\n\n"); continue; }

                id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
                if (!model) { printf("  FAIL: model\n\n"); continue; }

                id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
                NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
                NSFileManager *fm = [NSFileManager defaultManager];
                [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES attributes:nil error:nil];
                [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

                printf("  Compiling...\n");
                if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                    printf("  FAIL: compile — %s\n\n", [[e description] UTF8String]);
                    [fm removeItemAtPath:td error:nil];
                    continue;
                }
                printf("  Loading...\n");
                if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
                    printf("  FAIL: load — %s\n\n", [[e description] UTF8String]);
                    [fm removeItemAtPath:td error:nil];
                    continue;
                }
                printf("  Model ready!\n");

                // Input: [1, DIM, 1, SEQ] fp32 = DIM * SEQ * 4 bytes
                // Output: [1, VOCAB, 1, SEQ] fp32 = VOCAB * SEQ * 4 bytes
                NSUInteger inBytes = dim * seq * 4;
                NSUInteger outBytes = vocab * seq * 4;
                IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                    (id)kIOSurfaceWidth:@(inBytes),(id)kIOSurfaceHeight:@1,
                    (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(inBytes),
                    (id)kIOSurfaceAllocSize:@(inBytes),(id)kIOSurfacePixelFormat:@0});
                IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                    (id)kIOSurfaceWidth:@(outBytes),(id)kIOSurfaceHeight:@1,
                    (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(outBytes),
                    (id)kIOSurfaceAllocSize:@(outBytes),(id)kIOSurfacePixelFormat:@0});

                // Fill input
                IOSurfaceLock(ioIn, 0, NULL);
                float *in_ptr = (float*)IOSurfaceGetBaseAddress(ioIn);
                for (int i = 0; i < dim * seq; i++)
                    in_ptr[i] = (float)(arc4random() & 0xFFFF) / 65536.0f - 0.5f;
                IOSurfaceUnlock(ioIn, 0, NULL);

                id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
                id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

                // Warmup
                for (int i = 0; i < 5; i++) {
                    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                    if (!ok) {
                        printf("  FAIL: eval warmup — %s\n", e ? [[e description] UTF8String] : "unknown");
                        break;
                    }
                }

                int iters = 100;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                }
                double ane_ms = ticksToMs(mach_absolute_time() - t0) / iters;
                double gflops = 2.0 * seq * vocab * dim / 1e9;
                printf("  %d iters, avg: %.2f ms/eval\n", iters, ane_ms);
                printf("  TFLOPS: %.2f\n", gflops / (ane_ms / 1000) / 1000);

                // Read first output value
                IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                float *out_ptr = (float*)IOSurfaceGetBaseAddress(ioOut);
                printf("  First output: %.4f\n", out_ptr[0]);
                IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

                // Cleanup
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
                CFRelease(ioIn); CFRelease(ioOut);
                [fm removeItemAtPath:td error:nil];
            }
            printf("\n");
        }

        printf("=== DONE ===\n");
    }
    return 0;
}
