

#include <glog/logging.h>
#include <glog/log_severity.h>

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    google::InitGoogleLogging(argv[0]);

    google::SetStderrLogging(google::GLOG_WARNING);
    FLAGS_logtostderr = false; // 不输出到标准错误
    FLAGS_log_dir = "./test_log/"; // 指定日志文件存放目录
 
    // FLAGS_minloglevel = 0;
    FLAGS_logtostderr = true;
    int result = RUN_ALL_TESTS();

    google::ShutdownGoogleLogging();

    return result;
}


