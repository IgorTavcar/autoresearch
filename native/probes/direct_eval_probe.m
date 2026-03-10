// direct_eval_probe.m — Stage 1: doEvaluateDirectWithModel
// Compares standard ANE eval (through daemon XPC) vs direct eval (bypass daemon).
// Source: thebasedcapital/ane-infer
// Expected: ~10% latency reduction per dispatch (117μs → 106μs)
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToUs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Build a simple conv kernel MIL (768ch x 64sp — small enough to be fast)
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

        printf("=== Stage 1: doEvaluateDirectWithModel Probe ===\n");
        printf("Source: thebasedcapital/ane-infer\n");
        printf("Bypasses ANE daemon XPC for direct IOKit path\n\n");

        int ch = 768, sp = 64;
        NSError *e = nil;

        // Build and compile model
        NSData *milData = [[genMIL(ch, sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb = buildWeightBlob(ch);

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
        Class AR   = NSClassFromString(@"_ANERequest");
        Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}};
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!desc) { printf("FAIL: descriptor creation\n"); return 1; }

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
            printf("FAIL: compile — %s\n", [[e description] UTF8String]); return 1;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("FAIL: load — %s\n", [[e description] UTF8String]); return 1;
        }
        printf("Model compiled and loaded: %dch x %dsp\n\n", ch, sp);

        // Create IOSurfaces and request
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
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // =========================================
        // TEST 1: Standard eval (through daemon XPC)
        // =========================================
        printf("--- Standard evaluateWithQoS (through daemon) ---\n");

        // Warmup
        for (int i = 0; i < 20; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

        int iters = 1000;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double std_us = ticksToUs(mach_absolute_time() - t0) / iters;
        printf("  %d iterations, avg: %.1f μs/eval\n\n", iters, std_us);

        // =========================================
        // TEST 2: Direct eval via _ANEClient
        // =========================================
        printf("--- doEvaluateDirectWithModel (bypass daemon) ---\n");

        Class ANEClient = NSClassFromString(@"_ANEClient");
        id client = nil;
        id aneModel = nil;

        if (!ANEClient) {
            printf("FAIL: _ANEClient class not found\n");
        } else {
            // List class methods to find the right init
            printf("_ANEClient class methods:\n");
            unsigned int count;
            Method *methods = class_copyMethodList(object_getClass(ANEClient), &count);
            for (unsigned int i = 0; i < count; i++) {
                SEL s = method_getName(methods[i]);
                if (strstr(sel_getName(s), "init") || strstr(sel_getName(s), "alloc") || strstr(sel_getName(s), "shared"))
                    printf("  + %s\n", sel_getName(s));
            }
            free(methods);

            printf("_ANEClient instance methods (eval-related):\n");
            methods = class_copyMethodList(ANEClient, &count);
            for (unsigned int i = 0; i < count; i++) {
                SEL s = method_getName(methods[i]);
                if (strstr(sel_getName(s), "valuate") || strstr(sel_getName(s), "Direct") || strstr(sel_getName(s), "init"))
                    printf("  - %s\n", sel_getName(s));
            }
            free(methods);

            // Try creating a direct client
            if ([ANEClient instancesRespondToSelector:@selector(initWithRestrictedAccessAllowed:)]) {
                client = [[ANEClient alloc] init];
                client = ((id(*)(id,SEL,BOOL))objc_msgSend)(client, @selector(initWithRestrictedAccessAllowed:), YES);
                printf("Created _ANEClient with initWithRestrictedAccessAllowed:YES\n");
            } else {
                printf("initWithRestrictedAccessAllowed: not available, trying alternatives...\n");
                client = [[ANEClient alloc] init];
                if (client) printf("Created _ANEClient with plain init\n");
            }
        }

        if (client) {
            // Get the _ANEModel (kernel handle) from _ANEInMemoryModel
            // thebasedcapital says it's at offset +64
            @try {
                aneModel = [model valueForKey:@"model"];
                printf("Got _ANEModel via KVC: %s\n", aneModel ? [[aneModel description] UTF8String] : "nil");
            } @catch(NSException *ex) {
                printf("KVC 'model' failed, trying ivar offset +64...\n");
                Ivar ivar = class_getInstanceVariable([model class], "_model");
                if (ivar) {
                    aneModel = object_getIvar(model, ivar);
                    printf("Got _ANEModel via ivar: %s\n", aneModel ? [[aneModel description] UTF8String] : "nil");
                }
            }

            // Try doEvaluateDirectWithModel:options:request:qos:error:
            SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
            if ([client respondsToSelector:directSel]) {
                printf("\ndoEvaluateDirectWithModel: AVAILABLE!\n");

                // Warmup with direct path
                BOOL dok = NO;
                for (int i = 0; i < 20; i++) {
                    dok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, directSel, aneModel, @{}, req, 21, &e);
                }
                if (!dok) {
                    printf("Direct eval warmup with _ANEModel failed: %s\n", e ? [[e description] UTF8String] : "unknown");
                    // Try with the _ANEInMemoryModel instead
                    printf("Retrying with _ANEInMemoryModel...\n");
                    for (int i = 0; i < 5; i++) {
                        e = nil;
                        dok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, directSel, model, @{}, req, 21, &e);
                    }
                    if (!dok) {
                        printf("Also failed: %s\n", e ? [[e description] UTF8String] : "unknown");

                        // Try model's own direct eval method
                        SEL evalDirect2 = @selector(evaluateDirectWithQoS:options:request:error:);
                        if ([model respondsToSelector:evalDirect2]) {
                            printf("Trying evaluateDirectWithQoS on model object...\n");
                            for (int i = 0; i < 5; i++) {
                                e = nil;
                                dok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                                    model, evalDirect2, 21, @{}, req, &e);
                            }
                            printf("Result: %s\n", dok ? "SUCCESS" : (e ? [[e description] UTF8String] : "unknown"));
                        }
                    }
                }

                if (dok) {
                    printf("Direct eval warmup: SUCCESS\n");

                    // Benchmark direct path
                    t0 = mach_absolute_time();
                    for (int i = 0; i < iters; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, directSel, aneModel, @{}, req, 21, &e);
                    double direct_us = ticksToUs(mach_absolute_time() - t0) / iters;
                    printf("  %d iterations, avg: %.1f μs/eval\n\n", iters, direct_us);

                    printf("=== COMPARISON ===\n");
                    printf("Standard (XPC):  %.1f μs/eval\n", std_us);
                    printf("Direct (IOKit):  %.1f μs/eval\n", direct_us);
                    printf("Speedup:         %.1f%%\n", (1.0 - direct_us/std_us) * 100);
                    printf("Savings/eval:    %.1f μs\n", std_us - direct_us);
                    printf("Per step (144 dispatches): %.1f ms saved\n", (std_us - direct_us) * 144 / 1000);
                }
            } else {
                printf("doEvaluateDirectWithModel: NOT AVAILABLE on this _ANEClient\n");

                unsigned int count2;
                printf("\nAll _ANEClient instance methods:\n");
                Method *methods2 = class_copyMethodList(ANEClient, &count2);
                for (unsigned int i = 0; i < count2; i++)
                    printf("  - %s\n", sel_getName(method_getName(methods2[i])));
                free(methods2);
            }
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];

        printf("\n=== DONE ===\n");
    }
    return 0;
}
