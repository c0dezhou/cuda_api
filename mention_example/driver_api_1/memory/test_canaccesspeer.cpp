// 定义一个测试类
class cuDeviceCanAccessPeerTest : public ::testing::Test {
protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 初始化CUDA环境
    cuInit(0);
    // 获取可用的设备数量
    cuDeviceGetCount(&deviceCount);
    // 分配设备句柄数组
    devices = new CUdevice[deviceCount];
    // 获取每个设备的句柄
    for (int i = 0; i < deviceCount; i++) {
      cuDeviceGet(&devices[i], i);
    }
  }

  // 在每个测试之后执行
  void TearDown() override {
    // 释放设备句柄数组
    delete[] devices;
  }

  // 定义一些成员变量
  int deviceCount; // 设备数量
  CUdevice* devices; // 设备句柄数组
};

// 定义一个基本行为测试
TEST_F(cuDeviceCanAccessPeerTest, BasicBehavior) {
  // 定义一些变量
  int canAccessPeer; // 内存访问结果

  // 遍历所有的设备对
  for (int i = 0; i < deviceCount; i++) {
    for (int j = 0; j < deviceCount; j++) {
      // 调用cuDeviceCanAccessPeer函数，查询两个设备之间的内存访问能力
      cuDeviceCanAccessPeer(&canAccessPeer, devices[i], devices[j]);

      // 检查函数是否返回1或0
      EXPECT_TRUE(canAccessPeer == 1 || canAccessPeer == 0);

      // 如果返回1，打印设备对的信息
      if (canAccessPeer == 1) {
        std::cout << "Device " << i << " can access peer device " << j << std::endl;
      }
    }
  }
}



