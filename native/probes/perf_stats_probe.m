// perf_stats_probe.m — Stage 5: perfStatsMask + _ANEDeviceInfo
// Sets _perfStatsMask ivar on _ANEInMemoryModel to enable hardware perf counters.
// Also probes _ANEDeviceInfo for hardware detection.
// Source: thebasedcapital/ane-infer (ivar layout), Anemll (_ANEDeviceInfo)
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
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

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("=== Stage 5: perfStatsMask + _ANEDeviceInfo Probe ===\n\n");

        // =========================================
        // PART 1: _ANEDeviceInfo — Hardware Detection
        // =========================================
        printf("--- Part 1: _ANEDeviceInfo ---\n");
        Class DevInfo = NSClassFromString(@"_ANEDeviceInfo");
        if (DevInfo) {
            printf("_ANEDeviceInfo class found!\n");

            // List class methods
            unsigned int count;
            Method *methods = class_copyMethodList(object_getClass(DevInfo), &count);
            printf("Class methods:\n");
            for (unsigned int i = 0; i < count; i++)
                printf("  + %s\n", sel_getName(method_getName(methods[i])));
            free(methods);

            // Try known methods from Anemll
            @try {
                NSInteger numANEs = ((NSInteger(*)(Class,SEL))objc_msgSend)(DevInfo, @selector(numANEs));
                printf("numANEs: %ld\n", (long)numANEs);
            } @catch(NSException *ex) { printf("numANEs: not available\n"); }

            @try {
                NSInteger numCores = ((NSInteger(*)(Class,SEL))objc_msgSend)(DevInfo, @selector(numANECores));
                printf("numANECores: %ld\n", (long)numCores);
            } @catch(NSException *ex) { printf("numANECores: not available\n"); }

            @try {
                id subType = ((id(*)(Class,SEL))objc_msgSend)(DevInfo, @selector(aneSubType));
                printf("aneSubType: %s\n", subType ? [subType UTF8String] : "nil");
            } @catch(NSException *ex) { printf("aneSubType: not available\n"); }

            // Try any other methods
            @try {
                id info = ((id(*)(Class,SEL))objc_msgSend)(DevInfo, @selector(deviceInfo));
                printf("deviceInfo: %s\n", info ? [[info description] UTF8String] : "nil");
            } @catch(NSException *ex) { /* skip */ }
        } else {
            printf("_ANEDeviceInfo NOT FOUND\n");
        }

        // =========================================
        // PART 2: _ANEPerformanceStats — Perf Counter Class
        // =========================================
        printf("\n--- Part 2: _ANEPerformanceStats ---\n");
        Class PerfStats = NSClassFromString(@"_ANEPerformanceStats");
        if (PerfStats) {
            printf("_ANEPerformanceStats class found!\n");

            unsigned int count;
            Method *methods = class_copyMethodList(PerfStats, &count);
            printf("Instance methods:\n");
            for (unsigned int i = 0; i < count; i++) {
                SEL s = method_getName(methods[i]);
                const char *enc = method_getTypeEncoding(methods[i]);
                printf("  - %s [%s]\n", sel_getName(s), enc ? enc : "?");
            }
            free(methods);

            methods = class_copyMethodList(object_getClass(PerfStats), &count);
            printf("Class methods:\n");
            for (unsigned int i = 0; i < count; i++)
                printf("  + %s\n", sel_getName(method_getName(methods[i])));
            free(methods);
        } else {
            printf("_ANEPerformanceStats NOT FOUND\n");
        }

        // =========================================
        // PART 3: Compile model, set perfStatsMask, run eval
        // =========================================
        printf("\n--- Part 3: perfStatsMask on _ANEInMemoryModel ---\n");

        int ch = 768, sp = 64;
        NSError *e = nil;
        NSData *milData = [[genMIL(ch, sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb = buildWeightBlob(ch);

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
        Class AR   = NSClassFromString(@"_ANERequest");
        Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}};
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { printf("FAIL: model creation\n"); return 1; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("FAIL: compile\n"); return 1;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("FAIL: load\n"); return 1;
        }
        printf("Model compiled and loaded: %dch x %dsp\n", ch, sp);

        // Dump _ANEInMemoryModel ivar layout
        printf("\n_ANEInMemoryModel ivars:\n");
        {
            unsigned int ivarCount;
            Ivar *ivars = class_copyIvarList(IMM, &ivarCount);
            for (unsigned int i = 0; i < ivarCount; i++) {
                const char *name = ivar_getName(ivars[i]);
                ptrdiff_t offset = ivar_getOffset(ivars[i]);
                const char *type = ivar_getTypeEncoding(ivars[i]);
                printf("  [+%td] %s (%s)\n", offset, name ? name : "?", type ? type : "?");
            }
            free(ivars);
        }

        // Set _perfStatsMask at offset +12 (per thebasedcapital)
        printf("\nSetting _perfStatsMask = 0xFFFFFFFF at offset +12...\n");
        {
            // Try via KVC first
            @try {
                [model setValue:@(0xFFFFFFFF) forKey:@"perfStatsMask"];
                printf("Set via KVC: success\n");
            } @catch(NSException *ex) {
                printf("KVC failed (%s), trying direct memory...\n", [[ex reason] UTF8String]);
                // Direct memory write at offset +12
                uint32_t *mask = (uint32_t *)((uint8_t *)(__bridge void *)model + 12);
                printf("Current value at +12: 0x%08X\n", *mask);
                *mask = 0xFFFFFFFF;
                printf("Set to 0xFFFFFFFF via direct memory\n");
                printf("Verify: 0x%08X\n", *mask);
            }
        }

        // Create IOSurfaces
        NSUInteger bytes = ch * sp * 4;
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);

        // Create request WITH perfStats object
        id perfStatsObj = nil;
        if (PerfStats) {
            perfStatsObj = [[PerfStats alloc] init];
            printf("Created _ANEPerformanceStats object: %s\n", perfStatsObj ? "OK" : "FAIL");
        }

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, perfStatsObj, @0);
        printf("Request with perfStats: %s\n", req ? "created" : "FAIL");

        // Run eval
        printf("\nRunning eval with perfStats enabled...\n");
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("Eval result: %s\n", ok ? "SUCCESS" : "FAIL");

        // Read perf stats
        if (perfStatsObj) {
            printf("\nReading _ANEPerformanceStats after eval:\n");

            // Try all known properties
            SEL selectors[] = {
                @selector(hwExecutionTime),
                @selector(executionTime),
                @selector(totalTime),
                @selector(compileTime),
                @selector(loadTime),
                @selector(dmaTime),
                @selector(stallCycles),
                @selector(sramHits),
                @selector(description),
            };
            const char *names[] = {
                "hwExecutionTime", "executionTime", "totalTime",
                "compileTime", "loadTime", "dmaTime",
                "stallCycles", "sramHits", "description"
            };

            for (int i = 0; i < 9; i++) {
                @try {
                    if (i == 8) { // description returns NSString
                        id val = ((id(*)(id,SEL))objc_msgSend)(perfStatsObj, selectors[i]);
                        printf("  %s: %s\n", names[i], val ? [[val description] UTF8String] : "nil");
                    } else {
                        double val = ((double(*)(id,SEL))objc_msgSend)(perfStatsObj, selectors[i]);
                        printf("  %s: %.6f\n", names[i], val);
                    }
                } @catch(NSException *ex) {
                    printf("  %s: not available (%s)\n", names[i], [[ex reason] UTF8String]);
                }
            }

            // Also try integer selectors
            @try {
                uint64_t val = ((uint64_t(*)(id,SEL))objc_msgSend)(perfStatsObj, @selector(hwExecutionCycles));
                printf("  hwExecutionCycles: %llu\n", val);
            } @catch(NSException *ex) { printf("  hwExecutionCycles: not available\n"); }

            // Dump all properties via runtime
            printf("\nAll properties of _ANEPerformanceStats:\n");
            unsigned int propCount;
            objc_property_t *props = class_copyPropertyList(PerfStats, &propCount);
            for (unsigned int i = 0; i < propCount; i++) {
                const char *pname = property_getName(props[i]);
                const char *pattr = property_getAttributes(props[i]);
                printf("  @property %s [%s]\n", pname, pattr ? pattr : "?");

                // Try reading each property
                @try {
                    id val = [perfStatsObj valueForKey:[NSString stringWithUTF8String:pname]];
                    printf("    value: %s\n", val ? [[val description] UTF8String] : "nil/0");
                } @catch(NSException *ex) {
                    printf("    value: error\n");
                }
            }
            free(props);
        }

        // Run a few more evals and check if stats accumulate
        printf("\nRunning 10 more evals...\n");
        for (int i = 0; i < 10; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        }

        if (perfStatsObj) {
            printf("After 10 more evals:\n");
            printf("  description: %s\n", [[perfStatsObj description] UTF8String]);

            // Read all properties again
            unsigned int propCount;
            objc_property_t *props = class_copyPropertyList(PerfStats, &propCount);
            for (unsigned int i = 0; i < propCount; i++) {
                const char *pname = property_getName(props[i]);
                @try {
                    id val = [perfStatsObj valueForKey:[NSString stringWithUTF8String:pname]];
                    printf("  %s: %s\n", pname, val ? [[val description] UTF8String] : "nil/0");
                } @catch(NSException *ex) { /* skip */ }
            }
            free(props);
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];

        printf("\n=== DONE ===\n");
    }
    return 0;
}
