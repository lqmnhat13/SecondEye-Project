#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: secondeye-vision-ocr image\n");
            return 2;
        }
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];
        CGImageRef cgImage = [image CGImageForProposedRect:NULL context:nil hints:nil];
        if (cgImage == NULL) {
            fprintf(stderr, "cannot read image\n");
            return 3;
        }

        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.recognitionLanguages = @[@"vi-VN", @"en-US"];
        request.usesLanguageCorrection = YES;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
            initWithCGImage:cgImage options:@{}];
        CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
        NSError *error = nil;
        if (![handler performRequests:@[request] error:&error]) {
            const char *message = error.localizedDescription.UTF8String;
            fprintf(stderr, "%s\n", message == NULL ? "Vision request failed" : message);
            return 4;
        }

        NSMutableArray *lines = [NSMutableArray array];
        for (VNRecognizedTextObservation *observation in request.results) {
            VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
            if (candidate == nil) continue;
            CGRect box = observation.boundingBox;
            [lines addObject:@{
                @"text": candidate.string,
                @"confidence": @(candidate.confidence),
                @"box": @[@(CGRectGetMinX(box)), @(CGRectGetMinY(box)),
                           @(CGRectGetMaxX(box)), @(CGRectGetMaxY(box))],
            }];
        }
        NSDictionary *output = @{
            @"latency_ms": @((CFAbsoluteTimeGetCurrent() - started) * 1000.0),
            @"lines": lines,
        };
        NSData *data = [NSJSONSerialization dataWithJSONObject:output options:0 error:&error];
        if (data == nil) {
            const char *message = error.localizedDescription.UTF8String;
            fprintf(stderr, "%s\n", message == NULL ? "JSON encoding failed" : message);
            return 5;
        }
        fwrite(data.bytes, 1, data.length, stdout);
    }
    return 0;
}
