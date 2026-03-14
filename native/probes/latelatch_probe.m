// latelatch_probe.m — Stage 4: SkipPrepare + LateLatch + skipAdapterWeightAccessCheck
// Tests undocumented ANE options found in ANECompilerService binary.
// Source: Random X user (UNVERIFIED) — strings confirmed in dyld cache but never publicly used.
// Claims: model load drops from 20ms to 1.6ms
// CAUTION: These are unverified. Test output correctness carefully.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToUs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static NSString *genMIL(int ch, int sp) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    [m appendString:
        @"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        @"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", ch, sp];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n", ch, ch, ch, ch];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n", ch, sp];
    [m appendString:@"        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", ch, sp];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSData *buildWeightBlob(int ch) {
    NSUInteger wsize = (NSUInteger)ch * ch * 2;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf+72) = (uint32_t)wsize;
    *(uint32_t*)(buf+80) = 128;
    uint16_t *fp16 = (uint16_t*)(buf + 128);
    for (NSUInteger j = 0; j < (NSUInteger)ch * ch; j++)
        fp16[j] = (arc4random() & 0x03FF) | 0x2000;
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

typedef struct {
    id model;
    IOSurfaceRef ioIn, ioOut;
    id request;
    NSString *tmpDir;
} TestModel;

static TestModel createModel(int ch, int sp) {
    TestModel tm = {0};
    NSError *e = nil;
    NSData *milData = [[genMIL(ch, sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
    NSData *wb = buildWeightBlob(ch);

    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
    Class AR   = NSClassFromString(@"_ANERequest");
    Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

    NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}};
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
    tm.model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);

    id hx = ((id(*)(id,SEL))objc_msgSend)(tm.model, @selector(hexStringIdentifier));
    tm.tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tm.tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tm.tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wb writeToFile:[tm.tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(tm.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);

    NSUInteger bytes = ch * sp * 4;
    tm.ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
    tm.ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});

    // Fill input with deterministic data for correctness checking
    IOSurfaceLock(tm.ioIn, 0, NULL);
    float *ptr = (float*)IOSurfaceGetBaseAddress(tm.ioIn);
    for (int i = 0; i < ch * sp; i++) ptr[i] = 0.01f * (i % 100);
    IOSurfaceUnlock(tm.ioIn, 0, NULL);

    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), tm.ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), tm.ioOut);
    tm.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    return tm;
}

static void freeModel(TestModel *tm) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(tm->model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(tm->ioIn); CFRelease(tm->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:tm->tmpDir error:nil];
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("=== Stage 4: SkipPrepare + LateLatch Probe ===\n");
        printf("Source: Random X user (UNVERIFIED)\n");
        printf("Keys found in ANECompilerService binary via strings\n\n");

        int ch = 768, sp = 64;
        NSError *e = nil;
        int iters = 500;

        // Define option sets to test
        typedef struct {
            const char *name;
            NSDictionary *opts;
        } TestCase;

        TestCase cases[] = {
            {"empty options @{}", @{}},
            {"SkipPrepare only", @{@"kANEFSkipPreparePhaseKey": @YES}},
            {"LateLatch only", @{@"kANEFEnableLateLatchKey": @YES}},
            {"SkipPrepare + LateLatch", @{@"kANEFSkipPreparePhaseKey": @YES, @"kANEFEnableLateLatchKey": @YES}},
            {"skipAdapterWeightAccessCheck", @{@"ane_skipAdapterWeightAccessCheck": @YES}},
            {"All three", @{@"kANEFSkipPreparePhaseKey": @YES, @"kANEFEnableLateLatchKey": @YES, @"ane_skipAdapterWeightAccessCheck": @YES}},
        };
        int nCases = 6;

        // Get baseline output for correctness checking
        TestModel baseline = createModel(ch, sp);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(baseline.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            baseline.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, baseline.request, &e);

        float baseline_out[4];
        IOSurfaceLock(baseline.ioOut, kIOSurfaceLockReadOnly, NULL);
        memcpy(baseline_out, IOSurfaceGetBaseAddress(baseline.ioOut), sizeof(baseline_out));
        IOSurfaceUnlock(baseline.ioOut, kIOSurfaceLockReadOnly, NULL);
        printf("Baseline output[0:4]: %.6f %.6f %.6f %.6f\n\n", baseline_out[0], baseline_out[1], baseline_out[2], baseline_out[3]);
        freeModel(&baseline);

        // =========================================
        // TEST: evaluate with different options
        // =========================================
        printf("%-35s %10s %10s %s\n", "Options", "μs/eval", "vs base", "Correct?");
        printf("-------------------------------------------------------------------\n");

        double base_us = 0;

        for (int c = 0; c < nCases; c++) {
            TestModel tm = createModel(ch, sp);

            // Load with test options (some keys might apply to load too)
            BOOL loadOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm.model, @selector(loadWithQoS:options:error:), 21, cases[c].opts, &e);
            if (!loadOk) {
                printf("%-35s  FAIL (load: %s)\n", cases[c].name, e ? [[e description] UTF8String] : "unknown");
                [[NSFileManager defaultManager] removeItemAtPath:tm.tmpDir error:nil];
                CFRelease(tm.ioIn); CFRelease(tm.ioOut);
                continue;
            }

            // Warmup
            BOOL evalOk = YES;
            for (int i = 0; i < 20; i++) {
                e = nil;
                evalOk = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    tm.model, @selector(evaluateWithQoS:options:request:error:), 21, cases[c].opts, tm.request, &e);
                if (!evalOk) break;
            }
            if (!evalOk) {
                printf("%-35s  FAIL (eval: %s)\n", cases[c].name, e ? [[e description] UTF8String] : "unknown");
                freeModel(&tm);
                continue;
            }

            // Benchmark
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    tm.model, @selector(evaluateWithQoS:options:request:error:), 21, cases[c].opts, tm.request, &e);
            }
            double us = ticksToUs(mach_absolute_time() - t0) / iters;

            // Check correctness
            float test_out[4];
            IOSurfaceLock(tm.ioOut, kIOSurfaceLockReadOnly, NULL);
            memcpy(test_out, IOSurfaceGetBaseAddress(tm.ioOut), sizeof(test_out));
            IOSurfaceUnlock(tm.ioOut, kIOSurfaceLockReadOnly, NULL);

            bool correct = true;
            for (int i = 0; i < 4; i++) {
                float diff = fabsf(test_out[i] - baseline_out[i]);
                if (diff > 0.01f) { correct = false; break; }
            }

            if (c == 0) base_us = us;
            double speedup = (base_us > 0) ? (1.0 - us / base_us) * 100 : 0;
            printf("%-35s %9.1f  %+6.1f%%     %s\n", cases[c].name, us, speedup, correct ? "YES" : "NO!");

            freeModel(&tm);
        }

        // =========================================
        // TEST: load timing with different options
        // =========================================
        printf("\n\n--- Load timing with different options ---\n");
        printf("%-35s %10s %10s\n", "Options", "ms/load", "vs base");
        printf("---------------------------------------------------\n");

        double base_load_ms = 0;

        for (int c = 0; c < nCases; c++) {
            double total_ms = 0;
            int load_iters = 20;

            for (int li = 0; li < load_iters; li++) {
                TestModel tm = createModel(ch, sp);

                uint64_t t0 = mach_absolute_time();
                BOOL loadOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    tm.model, @selector(loadWithQoS:options:error:), 21, cases[c].opts, &e);
                total_ms += ticksToMs(mach_absolute_time() - t0);

                if (!loadOk) {
                    printf("%-35s  FAIL (load)\n", cases[c].name);
                    [[NSFileManager defaultManager] removeItemAtPath:tm.tmpDir error:nil];
                    CFRelease(tm.ioIn); CFRelease(tm.ioOut);
                    break;
                }

                freeModel(&tm);
            }

            double avg_ms = total_ms / load_iters;
            if (c == 0) base_load_ms = avg_ms;
            double speedup = (base_load_ms > 0) ? (1.0 - avg_ms / base_load_ms) * 100 : 0;
            printf("%-35s %9.2f  %+6.1f%%\n", cases[c].name, avg_ms, speedup);
        }

        printf("\n=== DONE ===\n");
    }
    return 0;
}
