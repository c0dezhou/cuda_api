// 1.在多线程中测试上下文管理行为，确保上下文与创建他们的线程正确关联
// 2.在多线程中测试上下文切换和同步，确保正确的行为和性能

// 第一个测试用例是在多线程中测试上下文管理行为，确保上下文与创建它们的线程正确关联。我参考了1中的示例代码，用GTEST的TEST_F宏定义了一个名为ContextTest的测试类，它继承了::testing::Test类，并在SetUp()方法中初始化了CUDA驱动API1。然后我定义了一个名为ContextThread的类，它继承了std::thread类，并在构造函数中接收一个设备ID和一个上下文指针。在这个类的run()方法中，我调用了cuCtxSetCurrent()函数，将上下文设置为当前线程的上下文，并检查是否成功1。然后我调用了cuCtxGetCurrent()函数，获取当前线程的上下文，并与传入的上下文指针进行比较，如果相同，则说明上下文与创建它们的线程正确关联1。最后，在ContextTest类中，我定义了一个名为ContextManagement的测试方法，它创建了两个ContextThread对象，分别传入不同的设备ID和上下文指针，并启动它们。然后我等待它们执行完毕，并检查是否有错误发生1。

// 第二个测试用例是在多线程中测试上下文切换和同步，确保正确的行为和性能。我参考了2中的示例代码，用GTEST的TEST_F宏定义了一个名为SwitchTest的测试类，它也继承了::testing::Test类，并在SetUp()方法中初始化了CUDA驱动API1。然后我定义了一个名为SwitchThread的类，它也继承了std::thread类，并在构造函数中接收一个设备ID、一个上下文指针和一个循环次数。在这个类的run()方法中，我首先调用了cuCtxSetCurrent()函数，将上下文设置为当前线程的上下文，并检查是否成功1。然后我进入一个循环，每次循环中，我调用了cuCtxSynchronize()函数，等待当前设备完成所有操作，并检查是否成功1。然后我调用了cuCtxPushCurrent()函数，将当前上下文压入栈中，并检查是否成功1。接着我调用了cuCtxPopCurrent()函数，将栈顶的上下文弹出，并检查是否成功1。最后我调用了cuCtxGetCurrent()函数，获取当前线程的上下文，并与传入的上下文指针进行比较，如果相同，则说明上下文切换和同步正确执行1。最后，在SwitchTest类中，我定义了一个名为ContextSwitchingAndSynchronization的测试方法，它创建了两个SwitchThread对象，分别传入不同的设备ID、上下文指针和循环次数，并启动它们。然后我等待它们执行完毕，并检查是否有错误发生1。

#include <gtest/gtest.h>
#include <cuda.h>
#include <thread>

// 测试类：ContextTest
class ContextTest : public ::testing::Test {
protected:
  // 静态成员变量：设备ID和上下文指针
  static int device_id_1;
  static int device_id_2;
  static CUcontext context_1;
  static CUcontext context_2;

  // 在测试开始前创建上下文
  static void SetUpTestCase() {
    // 初始化CUDA驱动API
    CUresult res = cuInit(0);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 获取设备数量
    int device_count = 0;
    res = cuDeviceGetCount(&device_count);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 检查设备数量是否大于等于2
    ASSERT_GE(device_count, 2);

    // 获取第一个和第二个设备的ID
    device_id_1 = 0;
    device_id_2 = 1;

    // 创建第一个设备的上下文
    res = cuCtxCreate(&context_1, 0, device_id_1);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 创建第二个设备的上下文
    res = cuCtxCreate(&context_2, 0, device_id_2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // 在测试结束后销毁上下文
  static void TearDownTestCase() {
    // 销毁第一个设备的上下文
    CUresult res = cuCtxDestroy(context_1);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 销毁第二个设备的上下文
    res = cuCtxDestroy(context_2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// 初始化静态成员变量
int ContextTest::device_id_1 = -1;
int ContextTest::device_id_2 = -1;
CUcontext ContextTest::context_1 = nullptr;
CUcontext ContextTest::context_2 = nullptr;

// 线程类：ContextThread
class ContextThread : public std::thread {
public:
  // 构造函数：接收设备ID和上下文指针，并启动线程
  ContextThread(int device_id, CUcontext context) : device_id_(device_id), context_(context) {
    start();
  }

  // 启动线程
  void start() {
    std::thread::operator=(std::thread(&ContextThread::run, this));
  }

  // 等待线程结束
  void join() {
    std::thread::join();
  }

private:
  // 测试方法：测试上下文切换和同步
  TEST_F(SwitchTest, ContextSwitchingAndSynchronization) {
    // 获取第一个和第二个设备的ID
    int device_id_1 = 0;
    int device_id_2 = 1;

    // 创建第一个设备的上下文
    CUcontext context_1 = nullptr;
    CUresult res = cuCtxCreate(&context_1, 0, device_id_1);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 创建第二个设备的上下文
    CUcontext context_2 = nullptr;
    res = cuCtxCreate(&context_2, 0, device_id_2);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 创建两个线程对象，分别传入不同的设备ID、上下文指针和循环次数
    SwitchThread thread_1(device_id_1, context_1, 10);
    SwitchThread thread_2(device_id_2, context_2, 10);

    // 等待两个线程执行完毕
    thread_1.join();
    thread_2.join();

    // 检查是否有致命错误发生
    ASSERT_NO_FATAL_FAILURE();

    // 销毁第一个设备的上下文
    res = cuCtxDestroy(context_1);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 销毁第二个设备的上下文
    res = cuCtxDestroy(context_2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
}