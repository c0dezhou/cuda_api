// 定义一个类型参数化测试夹具类
template <typename T>
class cuMemsetDTest : public ::testing::Test {
 protected:
  // 在每个测试开始之前执行
  void SetUp() override {
    // 分配设备内存
    size_t size = N * sizeof(T);
    cuMemAlloc(&d_ptr_, size);

    // 分配主机内存
    h_ptr_ = new T[N];
  }

  // 在每个测试结束之后执行
  void TearDown() override {
    // 释放内存
    delete[] h_ptr_;
    cuMemFree(d_ptr_);
  }

  // 成员变量
  CUdeviceptr d_ptr_; // 设备内存指针
  T* h_ptr_; // 主机内存指针
  static const size_t N = 100; // 元素个数
};

// 定义一个类型列表，包含要测试的数据类型
typedef ::testing::Types<unsigned char, unsigned short, unsigned int> MyTypes;

// 使用TYPED_TEST_SUITE宏注册类型参数化测试夹具类和类型列表
TYPED_TEST_SUITE(cuMemsetDTest, MyTypes);

// 使用TYPED_TEST宏定义不同数据类型的测试用例
TYPED_TEST(cuMemsetDTest, BasicBehavior) {
  // 获取当前的数据类型
  typedef TypeParam T;

  // 设置设备内存为最大值
  T val = std::numeric_limits<T>::max();
  cuMemsetD##sizeof(T)##Async(this->d_ptr_, val, this->N, NULL);

  // 同步设备
  cuDeviceSynchronize();

  // 拷贝设备内存到主机内存
  size_t size = this->N * sizeof(T);
  cuMemcpyDtoH(this->h_ptr_, this->d_ptr_, size);

  // 检查主机内存中的值是否都为最大值
  for (size_t i = 0; i < this->N; i++) {
    EXPECT_EQ(this->h_ptr_[i], val);
  }
}

// 其他的测试用例可以类似地定义，省略了细节



// another 操作
// 定义一个函数指针类型，用来表示不同的cuMemsetD*Async函数
typedef CUresult (*cuMemsetDAsyncFunc)(CUdeviceptr, unsigned int, size_t, CUstream);

// 定义一个测试夹具类，它可以接受一个函数指针作为参数
class cuMemsetDAsyncTest : public ::testing::TestWithParam<cuMemsetDAsyncFunc> {
 protected:
  // 在每个测试开始之前执行
  void SetUp() override {
    // 分配设备内存
    size_t size = N * sizeof(unsigned int);
    cuMemAlloc(&d_ptr_, size);

    // 分配主机内存
    h_ptr_ = new unsigned int[N];
  }

  // 在每个测试结束之后执行
  void TearDown() override {
    // 释放内存
    delete[] h_ptr_;
    cuMemFree(d_ptr_);
  }

  // 成员变量
  CUdeviceptr d_ptr_; // 设备内存指针
  unsigned int* h_ptr_; // 主机内存指针
  static const size_t N = 100; // 元素个数
};

// 定义一个函数，用来返回不同的cuMemsetD*Async函数指针
cuMemsetDAsyncFunc GetCuMemsetDAsyncFunc(size_t size) {
  switch (size) {
    case 1:
      return &cuMemsetD8Async;
    case 2:
      return &cuMemsetD16Async;
    case 4:
      return &cuMemsetD32Async;
    default:
      return nullptr;
  }
}

// 定义一个函数，用来生成不同的cuMemsetD*Async函数指针作为参数
std::vector<cuMemsetDAsyncFunc> GenerateParams() {
  std::vector<cuMemsetDAsyncFunc> params;
  for (size_t size = 1; size <= 4; size *= 2) {
    params.push_back(GetCuMemsetDAsyncFunc(size));
  }
  return params;
}

// 使用INSTANTIATE_TEST_SUITE_P宏注册测试夹具类和参数列表
INSTANTIATE_TEST_SUITE_P(cuMemsetDAsyncTest, cuMemsetDAsyncTest,
                         ::testing::ValuesIn(GenerateParams()));

// 使用TEST_P宏定义不同函数指针的测试用例
TEST_P(cuMemsetDAsyncTest, BasicBehavior) {
  // 获取当前的函数指针
  cuMemsetDAsyncFunc func = GetParam();

  // 设置设备内存为最大值
  unsigned int val = std::numeric_limits<unsigned int>::max();
  func(this->d_ptr_, val, this->N, NULL);

  // 同步设备
  cuDeviceSynchronize();

  // 拷贝设备内存到主机内存
  size_t size = this->N * sizeof(unsigned int);
  cuMemcpyDtoH(this->h_ptr_, this->d_ptr_, size);

  // 检查主机内存中的值是否都为最大值
  for (size_t i = 0; i < this->N; i++) {
    EXPECT_EQ(this->h_ptr_[i], val);
  }
}

// 其他的测试用例可以类似地定义，省略了细节
