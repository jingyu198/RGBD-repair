#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ImageMsg = sensor_msgs::msg::Image;
using Clock = std::chrono::steady_clock;

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            std::ostringstream oss;                                           \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(status);                                \
            throw std::runtime_error(oss.str());                              \
        }                                                                     \
    } while (0)

double msBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double meanMs(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open engine: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    file.read(data.data(), size);
    if (!file) {
        throw std::runtime_error("failed to read engine: " + path);
    }
    return data;
}

size_t dtypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kHALF:
        case nvinfer1::DataType::kBF16:
            return 2;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kFP8:
        case nvinfer1::DataType::kINT4:
            return 1;
        case nvinfer1::DataType::kINT64:
            return 8;
        default:
            throw std::runtime_error("unsupported TensorRT dtype");
    }
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("tensor shape is not fully specified");
        }
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

bool hasDynamicDim(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims defaultInputShape(const std::string& name, int h, int w) {
    nvinfer1::Dims dims{};
    if (name.find("image") != std::string::npos) {
        dims.nbDims = 4;
        dims.d[0] = 1;
        dims.d[1] = 3;
        dims.d[2] = h;
        dims.d[3] = w;
    } else {
        dims.nbDims = 3;
        dims.d[0] = 1;
        dims.d[1] = h;
        dims.d[2] = w;
    }
    return dims;
}

template <typename T>
T readPixel(const uint8_t* base, size_t offset) {
    T value{};
    std::memcpy(&value, base + offset, sizeof(T));
    return value;
}

cv::Mat imageToBgr(const ImageMsg& msg) {
    if (msg.height == 0 || msg.width == 0) {
        throw std::runtime_error("empty RGB image");
    }
    const uint8_t* data = msg.data.data();
    if (msg.encoding == "bgr8" || msg.encoding == "rgb8") {
        cv::Mat view(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
                     const_cast<uint8_t*>(data), msg.step);
        cv::Mat out = view.clone();
        if (msg.encoding == "rgb8") {
            cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
        }
        return out;
    }
    if (msg.encoding == "bgra8" || msg.encoding == "rgba8") {
        cv::Mat view(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4,
                     const_cast<uint8_t*>(data), msg.step);
        cv::Mat out;
        cv::cvtColor(view, out, msg.encoding == "bgra8" ? cv::COLOR_BGRA2BGR : cv::COLOR_RGBA2BGR);
        return out;
    }
    throw std::runtime_error("unsupported RGB encoding: " + msg.encoding);
}

cv::Mat imageToDepthMeters(const ImageMsg& msg) {
    if (msg.height == 0 || msg.width == 0) {
        throw std::runtime_error("empty depth image");
    }
    cv::Mat out(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_32FC1);
    const uint8_t* base = msg.data.data();
    if (msg.encoding == "16UC1" || msg.encoding == "TYPE_16UC1" || msg.encoding == "mono16" ||
        msg.encoding == "MONO16") {
        for (uint32_t y = 0; y < msg.height; ++y) {
            float* dst = out.ptr<float>(static_cast<int>(y));
            const uint8_t* row = base + static_cast<size_t>(y) * msg.step;
            for (uint32_t x = 0; x < msg.width; ++x) {
                const uint16_t raw = readPixel<uint16_t>(row, static_cast<size_t>(x) * sizeof(uint16_t));
                dst[x] = (raw == 0 || raw == 65535) ? 0.0f : static_cast<float>(raw) * 0.001f;
            }
        }
        return out;
    }
    if (msg.encoding == "32FC1" || msg.encoding == "TYPE_32FC1") {
        for (uint32_t y = 0; y < msg.height; ++y) {
            float* dst = out.ptr<float>(static_cast<int>(y));
            const uint8_t* row = base + static_cast<size_t>(y) * msg.step;
            for (uint32_t x = 0; x < msg.width; ++x) {
                const float raw = readPixel<float>(row, static_cast<size_t>(x) * sizeof(float));
                dst[x] = (std::isfinite(raw) && raw > 0.0f) ? raw : 0.0f;
            }
        }
        return out;
    }
    throw std::runtime_error("unsupported depth encoding: " + msg.encoding);
}

void prepareInputs(const cv::Mat& colorBgr, const cv::Mat& depthM, int outW, int outH,
                   std::vector<float>& image, std::vector<float>& depth) {
    cv::Mat colorSmall;
    cv::resize(colorBgr, colorSmall, cv::Size(outW, outH), 0, 0, cv::INTER_AREA);
    const int hw = outW * outH;
    for (int y = 0; y < outH; ++y) {
        const cv::Vec3b* row = colorSmall.ptr<cv::Vec3b>(y);
        for (int x = 0; x < outW; ++x) {
            const int i = y * outW + x;
            image[i] = static_cast<float>(row[x][2]) / 255.0f;
            image[hw + i] = static_cast<float>(row[x][1]) / 255.0f;
            image[2 * hw + i] = static_cast<float>(row[x][0]) / 255.0f;
        }
    }

    cv::Mat depthSmall;
    cv::resize(depthM, depthSmall, cv::Size(outW, outH), 0, 0, cv::INTER_NEAREST);
    for (int y = 0; y < outH; ++y) {
        const float* row = depthSmall.ptr<float>(y);
        for (int x = 0; x < outW; ++x) {
            const float v = row[x];
            depth[y * outW + x] = (std::isfinite(v) && v > 0.0f) ? v : 0.0f;
        }
    }
}

struct TensorBuffer {
    std::string name;
    nvinfer1::TensorIOMode mode{};
    nvinfer1::Dims shape{};
    nvinfer1::DataType dtype{};
    void* device = nullptr;
    size_t bytes = 0;
};

class TrtRunner {
public:
    TrtRunner(const std::string& enginePath, int outW, int outH) : outW_(outW), outH_(outH) {
        CHECK_CUDA(cudaSetDevice(0));
        auto engineData = readFile(enginePath);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine_) {
            throw std::runtime_error("failed to deserialize TensorRT engine");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("failed to create TensorRT execution context");
        }
        std::vector<std::string> tensorNames;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            tensorNames.emplace_back(engine_->getIOTensorName(i));
        }
        for (const auto& name : tensorNames) {
            if (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT &&
                hasDynamicDim(engine_->getTensorShape(name.c_str()))) {
                if (!context_->setInputShape(name.c_str(), defaultInputShape(name, outH_, outW_))) {
                    throw std::runtime_error("failed to set input shape: " + name);
                }
            }
        }
        for (const auto& name : tensorNames) {
            TensorBuffer b;
            b.name = name;
            b.mode = engine_->getTensorIOMode(name.c_str());
            b.shape = context_->getTensorShape(name.c_str());
            b.dtype = engine_->getTensorDataType(name.c_str());
            b.bytes = volume(b.shape) * dtypeSize(b.dtype);
            CHECK_CUDA(cudaMalloc(&b.device, b.bytes));
            if (!context_->setTensorAddress(b.name.c_str(), b.device)) {
                throw std::runtime_error("failed to bind tensor: " + b.name);
            }
            if (b.mode == nvinfer1::TensorIOMode::kINPUT && b.name.find("image") != std::string::npos) {
                imageDevice_ = b.device;
            } else if (b.mode == nvinfer1::TensorIOMode::kINPUT) {
                depthDevice_ = b.device;
            } else {
                if (b.dtype != nvinfer1::DataType::kFLOAT) {
                    throw std::runtime_error("expected FP32 output tensor");
                }
                outputDevice_ = b.device;
                outputBytes_ = b.bytes;
            }
            buffers_.push_back(b);
        }
        if (!imageDevice_ || !depthDevice_ || !outputDevice_) {
            throw std::runtime_error("unexpected TensorRT bindings");
        }
        CHECK_CUDA(cudaStreamCreate(&stream_));
        CHECK_CUDA(cudaEventCreate(&startEvent_));
        CHECK_CUDA(cudaEventCreate(&stopEvent_));
    }

    ~TrtRunner() {
        for (auto& b : buffers_) {
            if (b.device) {
                cudaFree(b.device);
            }
        }
        if (startEvent_) {
            cudaEventDestroy(startEvent_);
        }
        if (stopEvent_) {
            cudaEventDestroy(stopEvent_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    float infer(const std::vector<float>& image, const std::vector<float>& depth, std::vector<float>& output) {
        output.resize(outputBytes_ / sizeof(float));
        CHECK_CUDA(cudaEventRecord(startEvent_, stream_));
        CHECK_CUDA(cudaMemcpyAsync(imageDevice_, image.data(), image.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        CHECK_CUDA(cudaMemcpyAsync(depthDevice_, depth.data(), depth.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
        CHECK_CUDA(cudaMemcpyAsync(output.data(), outputDevice_, outputBytes_, cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaEventRecord(stopEvent_, stream_));
        CHECK_CUDA(cudaEventSynchronize(stopEvent_));
        float trtMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&trtMs, startEvent_, stopEvent_));
        return trtMs;
    }

private:
    int outW_ = 64;
    int outH_ = 36;
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<TensorBuffer> buffers_;
    void* imageDevice_ = nullptr;
    void* depthDevice_ = nullptr;
    void* outputDevice_ = nullptr;
    size_t outputBytes_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t startEvent_ = nullptr;
    cudaEvent_t stopEvent_ = nullptr;
};

double stampSec(const builtin_interfaces::msg::Time& stamp) {
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1.0e-9;
}

class LingBotDepthNode final : public rclcpp::Node {
public:
    LingBotDepthNode()
        : Node("a3_lingbot_depth_trt_node") {
        const std::string enginePath = declare_parameter<std::string>(
            "engine_path", "trt/lingbot_tokens128_36x64_int8.engine");
        rgbTopic_ = declare_parameter<std::string>("rgb_topic", "/hal/chest_front_d457_camera/rgb");
        depthTopic_ = declare_parameter<std::string>("depth_topic", "/hal/chest_front_d457_camera/depth");
        publishTopic_ = declare_parameter<std::string>("publish_topic", "/hal/chest_front_d457_camera/depth_lingbot");
        outW_ = declare_parameter<int>("out_width", 64);
        outH_ = declare_parameter<int>("out_height", 36);
        maxDepthM_ = declare_parameter<double>("max_depth_m", 50.0);
        maxRgbAgeMs_ = declare_parameter<double>("max_rgb_age_ms", 150.0);
        printEverySec_ = declare_parameter<double>("print_every_sec", 1.0);
        const int qosDepth = declare_parameter<int>("qos_depth", 2);

        runner_ = std::make_unique<TrtRunner>(enginePath, outW_, outH_);

        auto inputQos = rclcpp::QoS(rclcpp::KeepLast(std::max(1, qosDepth))).best_effort().durability_volatile();
        auto outputQos = rclcpp::QoS(rclcpp::KeepLast(std::max(1, qosDepth))).reliable().durability_volatile();
        rgbSub_ = create_subscription<ImageMsg>(rgbTopic_, inputQos,
                                                std::bind(&LingBotDepthNode::handleRgb, this, std::placeholders::_1));
        depthSub_ = create_subscription<ImageMsg>(depthTopic_, inputQos,
                                                  std::bind(&LingBotDepthNode::handleDepth, this, std::placeholders::_1));
        pub_ = create_publisher<ImageMsg>(publishTopic_, outputQos);
        statStart_ = Clock::now();
        RCLCPP_INFO(get_logger(),
                    "LingBot TensorRT relay ready: rgb=%s depth=%s -> %s, engine=%s, output=%dx%d, no metric calibration",
                    rgbTopic_.c_str(), depthTopic_.c_str(), publishTopic_.c_str(), enginePath.c_str(), outW_, outH_);
    }

private:
    void handleRgb(const ImageMsg::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(rgbMutex_);
        latestRgb_ = msg;
    }

    void handleDepth(const ImageMsg::SharedPtr depthMsg) {
        const auto frameStart = Clock::now();
        ImageMsg::SharedPtr rgbMsg;
        {
            std::lock_guard<std::mutex> lock(rgbMutex_);
            rgbMsg = latestRgb_;
        }
        if (!rgbMsg) {
            ++dropCount_;
            return;
        }
        if (maxRgbAgeMs_ > 0.0) {
            const double diffMs = std::abs(stampSec(depthMsg->header.stamp) - stampSec(rgbMsg->header.stamp)) * 1000.0;
            if (diffMs > maxRgbAgeMs_) {
                ++dropCount_;
                return;
            }
        }

        try {
            const auto prepStart = Clock::now();
            cv::Mat rgbBgr = imageToBgr(*rgbMsg);
            cv::Mat depthM = imageToDepthMeters(*depthMsg);
            std::vector<float> image(static_cast<size_t>(3 * outW_ * outH_));
            std::vector<float> depth(static_cast<size_t>(outW_ * outH_));
            prepareInputs(rgbBgr, depthM, outW_, outH_, image, depth);
            const auto prepEnd = Clock::now();

            std::vector<float> output;
            const float trtMs = runner_->infer(image, depth, output);

            ImageMsg out;
            out.header = depthMsg->header;
            out.height = static_cast<uint32_t>(outH_);
            out.width = static_cast<uint32_t>(outW_);
            out.encoding = "32FC1";
            out.is_bigendian = false;
            out.step = static_cast<uint32_t>(outW_ * sizeof(float));
            out.data.resize(static_cast<size_t>(outW_ * outH_) * sizeof(float));
            auto* dst = reinterpret_cast<float*>(out.data.data());
            const size_t n = std::min(output.size(), static_cast<size_t>(outW_ * outH_));
            for (size_t i = 0; i < n; ++i) {
                float v = output[i];
                if (!std::isfinite(v) || v < 0.0f) {
                    v = 0.0f;
                }
                if (maxDepthM_ > 0.0) {
                    v = std::min(v, static_cast<float>(maxDepthM_));
                }
                dst[i] = v;
            }
            for (size_t i = n; i < static_cast<size_t>(outW_ * outH_); ++i) {
                dst[i] = 0.0f;
            }
            pub_->publish(out);

            const auto frameEnd = Clock::now();
            ++frameCount_;
            prepMs_.push_back(msBetween(prepStart, prepEnd));
            trtMs_.push_back(static_cast<double>(trtMs));
            totalMs_.push_back(msBetween(frameStart, frameEnd));
            while (prepMs_.size() > 120) {
                prepMs_.pop_front();
                trtMs_.pop_front();
                totalMs_.pop_front();
            }
            logStats(frameEnd);
        } catch (const std::exception& e) {
            ++dropCount_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "LingBot dropped frame: %s", e.what());
        }
    }

    void logStats(Clock::time_point now) {
        if (printEverySec_ <= 0.0 || msBetween(statStart_, now) < printEverySec_ * 1000.0) {
            return;
        }
        const double elapsedSec = msBetween(statStart_, now) / 1000.0;
        const uint64_t frames = frameCount_ - statFrameStart_;
        RCLCPP_INFO(get_logger(),
                    "publish_fps=%.2f frames=%lu dropped=%lu trt_ms=%.3f prep_ms=%.3f total_ms=%.3f",
                    static_cast<double>(frames) / std::max(elapsedSec, 1.0e-6),
                    static_cast<unsigned long>(frameCount_),
                    static_cast<unsigned long>(dropCount_),
                    meanMs(trtMs_),
                    meanMs(prepMs_),
                    meanMs(totalMs_));
        statStart_ = now;
        statFrameStart_ = frameCount_;
    }

    std::string rgbTopic_;
    std::string depthTopic_;
    std::string publishTopic_;
    int outW_ = 64;
    int outH_ = 36;
    double maxDepthM_ = 50.0;
    double maxRgbAgeMs_ = 150.0;
    double printEverySec_ = 1.0;
    std::unique_ptr<TrtRunner> runner_;
    rclcpp::Subscription<ImageMsg>::SharedPtr rgbSub_;
    rclcpp::Subscription<ImageMsg>::SharedPtr depthSub_;
    rclcpp::Publisher<ImageMsg>::SharedPtr pub_;
    std::mutex rgbMutex_;
    ImageMsg::SharedPtr latestRgb_;
    uint64_t frameCount_ = 0;
    uint64_t dropCount_ = 0;
    uint64_t statFrameStart_ = 0;
    Clock::time_point statStart_;
    std::deque<double> prepMs_;
    std::deque<double> trtMs_;
    std::deque<double> totalMs_;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<LingBotDepthNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
