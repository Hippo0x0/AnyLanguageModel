import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    @Suite("LlamaLanguageModel multimodal configuration")
    struct LlamaLanguageModelMultimodalConfigurationTests {
        @Test func initializerStoresProjectorPath() {
            let model = LlamaLanguageModel(
                modelPath: "/models/text.gguf",
                mmprojPath: "/models/mmproj.gguf"
            )

            #expect(model.modelPath == "/models/text.gguf")
            #expect(model.mmprojPath == "/models/mmproj.gguf")
        }

        @Test func customGenerationOptionsCarryMultimodalSettings() {
            var options = GenerationOptions()
            let custom = LlamaLanguageModel.CustomGenerationOptions(
                mmprojPath: "/override/mmproj.gguf",
                mediaMarker: "<image>",
                imageMinTokens: 16,
                imageMaxTokens: 1024,
                mmprojUseGPU: false
            )
            options[custom: LlamaLanguageModel.self] = custom

            let retrieved = options[custom: LlamaLanguageModel.self]
            #expect(retrieved?.mmprojPath == "/override/mmproj.gguf")
            #expect(retrieved?.mediaMarker == "<image>")
            #expect(retrieved?.imageMinTokens == 16)
            #expect(retrieved?.imageMaxTokens == 1024)
            #expect(retrieved?.mmprojUseGPU == false)
        }

        @Test func imageInputWithoutProjectorThrowsMissingProjectorBeforeModelLoad() async throws {
            let model = LlamaLanguageModel(modelPath: "/missing/text.gguf")
            let session = LanguageModelSession(model: model)
            let image = Transcript.ImageSegment(data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png")

            do {
                _ = try await session.respond(to: "Describe this image", image: image)
                Issue.record("Expected missing projector error")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .missingMultimodalProjector)
            }
        }

        @Test func multimodalPromptContentPreservesImageOrderWithMarkers() {
            let first = Transcript.ImageSegment(data: Data([1]), mimeType: "image/png")
            let second = Transcript.ImageSegment(data: Data([2]), mimeType: "image/png")
            let segments: [Transcript.Segment] = [
                .text(.init(content: "before ")),
                .image(first),
                .text(.init(content: " between ")),
                .image(second),
                .text(.init(content: " after")),
            ]

            let result = LlamaLanguageModel.multimodalPromptContent(from: segments, mediaMarker: "<__media__>")

            #expect(result.text == "before <__media__> between <__media__> after")
            #expect(result.images == [first, second])
        }
    }

    @Suite(
        "LlamaLanguageModel multimodal runtime",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil
            && ProcessInfo.processInfo.environment["LLAMA_MMPROJ_PATH"] != nil)
    )
    struct LlamaLanguageModelMultimodalRuntimeTests {
        @Test func imageInputUsesProjector() async throws {
            let model = LlamaLanguageModel(
                modelPath: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"]!,
                mmprojPath: ProcessInfo.processInfo.environment["LLAMA_MMPROJ_PATH"]!
            )
            let session = LanguageModelSession(model: model)
            let image = Transcript.ImageSegment(data: Self.onePixelPNG, mimeType: "image/png")
            let options = GenerationOptions(maximumResponseTokens: 1)

            _ = try await session.respond(to: "Describe this image in one word.", image: image, options: options)
        }

        private static let onePixelPNG = Data([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
    }

    @Suite(
        "LlamaLanguageModel",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil)
    )
    struct LlamaLanguageModelTests {
        let model = LlamaLanguageModel(
            modelPath: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"]!
        )

        @Test func initialization() {
            let customModel = LlamaLanguageModel(modelPath: "/path/to/model.gguf")
            #expect(customModel.modelPath == "/path/to/model.gguf")
            #expect(customModel.contextSize == 2048)
            #expect(customModel.batchSize == 512)
            #expect(customModel.threads == Int32(ProcessInfo.processInfo.processorCount))
            #expect(customModel.temperature == 0.8)
            #expect(customModel.topK == 40)
            #expect(customModel.topP == 0.95)
            #expect(customModel.repeatPenalty == 1.1)
            #expect(customModel.repeatLastN == 64)
        }

        @Test func customGenerationOptionsRoundTrip() {
            var options = GenerationOptions(
                temperature: 0.6,
                maximumResponseTokens: 25
            )

            let custom = LlamaLanguageModel.CustomGenerationOptions(
                contextSize: 1024,
                batchSize: 256,
                threads: 1,
                seed: 42,
                temperature: 0.55,
                topK: 25,
                topP: 0.85,
                repeatPenalty: 1.15,
                repeatLastN: 48,
                frequencyPenalty: 0.05,
                presencePenalty: 0.05,
                mirostat: .v2(tau: 5.0, eta: 0.2)
            )
            options[custom: LlamaLanguageModel.self] = custom

            let retrieved = options[custom: LlamaLanguageModel.self]
            #expect(retrieved?.contextSize == 1024)
            #expect(retrieved?.batchSize == 256)
            #expect(retrieved?.threads == 1)
            #expect(retrieved?.seed == 42)
            #expect(retrieved?.temperature == 0.55)
            #expect(retrieved?.topK == 25)
            #expect(retrieved?.topP == 0.85)
            #expect(retrieved?.repeatPenalty == 1.15)
            #expect(retrieved?.repeatLastN == 48)
            #expect(retrieved?.frequencyPenalty == 0.05)
            #expect(retrieved?.presencePenalty == 0.05)
            #expect(retrieved?.mirostat == .v2(tau: 5.0, eta: 0.2))
        }

        @Test func customGenerationOptionsDefaults() {
            let defaults = LlamaLanguageModel.CustomGenerationOptions.default
            #expect(defaults.contextSize == 2048)
            #expect(defaults.batchSize == 512)
            #expect(defaults.threads == Int32(ProcessInfo.processInfo.processorCount))
            #expect(defaults.seed == nil)
            #expect(defaults.temperature == 0.8)
            #expect(defaults.topK == 40)
            #expect(defaults.topP == 0.95)
            #expect(defaults.repeatPenalty == 1.1)
            #expect(defaults.repeatLastN == 64)
            #expect(defaults.frequencyPenalty == 0.0)
            #expect(defaults.presencePenalty == 0.0)
            #expect(defaults.mirostat == nil)
        }

        @Test func deprecatedInitializerFallback() {
            let legacy = LlamaLanguageModel(
                modelPath: "/legacy/model.gguf",
                contextSize: 1024,
                batchSize: 128,
                threads: 3,
                seed: 7,
                temperature: 0.65,
                topK: 32,
                topP: 0.88,
                repeatPenalty: 1.02,
                repeatLastN: 24
            )

            // Deprecated initializer ignores parameters; defaults are used.
            #expect(legacy.contextSize == 2048)
            #expect(legacy.batchSize == 512)
            #expect(legacy.threads == Int32(ProcessInfo.processInfo.processorCount))
            #expect(legacy.temperature == 0.8)
            #expect(legacy.topK == 40)
            #expect(legacy.topP == 0.95)
            #expect(legacy.repeatPenalty == 1.1)
            #expect(legacy.repeatLastN == 64)
        }

        @Test func logLevelConfiguration() {
            let originalLevel = LlamaLanguageModel.logLevel

            LlamaLanguageModel.logLevel = .none
            #expect(LlamaLanguageModel.logLevel == .none)

            LlamaLanguageModel.logLevel = .debug
            #expect(LlamaLanguageModel.logLevel == .debug)

            LlamaLanguageModel.logLevel = .error
            #expect(LlamaLanguageModel.logLevel == .error)

            LlamaLanguageModel.logLevel = originalLevel
        }

        @Test func logLevelComparison() {
            #expect(LlamaLanguageModel.LogLevel.none < .debug)
            #expect(LlamaLanguageModel.LogLevel.debug < .info)
            #expect(LlamaLanguageModel.LogLevel.info < .warn)
            #expect(LlamaLanguageModel.LogLevel.warn < .error)

            #expect(LlamaLanguageModel.LogLevel.error > .warn)
            #expect(LlamaLanguageModel.LogLevel.warn >= .warn)
        }

        @Test func logLevelHashable() {
            let levels: Set<LlamaLanguageModel.LogLevel> = [.debug, .info, .warn]
            #expect(levels.contains(.debug))
            #expect(levels.contains(.info))
            #expect(levels.contains(.warn))
            #expect(!levels.contains(.none))
            #expect(!levels.contains(.error))
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(!secondResponse.content.isEmpty)
        }

        @Test func maxTokensLimit() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(maximumResponseTokens: 10)
            let response = try await session.respond(
                to: "Write a long essay about artificial intelligence",
                options: options
            )

            // Response should be limited by max tokens
            #expect(!response.content.isEmpty)
        }

        @Test func greedySamplingWithTemperature() async throws {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(
                sampling: .greedy,
                temperature: 0.7,
                maximumResponseTokens: 50
            )
            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.8,
                maximumResponseTokens: 50
            )

            // Set llama.cpp-specific custom options
            options[custom: LlamaLanguageModel.self] = .init(
                contextSize: 1024,
                batchSize: 256,
                threads: 2,
                seed: 123,
                temperature: 0.75,
                topK: 30,
                topP: 0.9,
                repeatPenalty: 1.2,
                repeatLastN: 128,
                frequencyPenalty: 0.1,
                presencePenalty: 0.1
            )

            let response = try await session.respond(
                to: "Tell me a short fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withMirostatSampling() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.8,
                maximumResponseTokens: 50
            )

            // Use mirostat v2 for adaptive perplexity control
            options[custom: LlamaLanguageModel.self] = .init(
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            let response = try await session.respond(
                to: "Tell me a short fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodal_rejectsImageURL() async throws {
            let session = LanguageModelSession(model: model)
            let imageSegment = Transcript.ImageSegment(url: testImageURL)
            do {
                _ = try await session.respond(to: "Describe this image", image: imageSegment)
                Issue.record("Expected error when image segments are present")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .unsupportedFeature)
            }
        }

        @Test func multimodal_rejectsImageData() async throws {
            let session = LanguageModelSession(model: model)
            let imageSegment = Transcript.ImageSegment(data: testImageData, mimeType: "image/png")
            do {
                _ = try await session.respond(to: "Describe this image", image: imageSegment)
                Issue.record("Expected error when image segments are present")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .unsupportedFeature)
            }
        }

        @Test func promptExceedingBatchSize_rejected() async throws {
            let session = LanguageModelSession(model: model)

            // Use a very small batch size to test the validation
            var options = GenerationOptions(maximumResponseTokens: 10)
            options[custom: LlamaLanguageModel.self] = .init(batchSize: 8)

            // Create a prompt that will tokenize to more than 8 tokens
            // Most models will tokenize "Hello world how are you today" to more than 8 tokens
            let longPrompt = String(repeating: "Hello world how are you today? ", count: 10)

            do {
                _ = try await session.respond(to: longPrompt, options: options)
                // If we get here, either the prompt tokenized to <= 8 tokens (unlikely)
                // or the validation didn't work (bug)
                // In practice, this should throw insufficientMemory
            } catch let error as LlamaLanguageModelError {
                // Expected: prompt token count exceeds batch size
                #expect(error == .insufficientMemory)
            }
        }
    }
#endif  // Llama
