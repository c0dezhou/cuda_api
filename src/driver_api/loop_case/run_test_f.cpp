// A code to loop each module wrapped by TEST_F() only when needed
#include <iostream>
#include <gtest/gtest.h>

// A helper macro to loop 100 times over a function call and check the result
#define LOOP_100(func) \
  do { \
    for (int i = 0; i < 100; i++) { \
      ASSERT_TRUE(func); \
    } \
  } while (0)

// A helper macro to print a message before and after a function call
#define PRINT_MSG(func) \
  do { \
    std::cout << "Calling " << #func << std::endl; \
    func; \
    std::cout << "Done " << #func << std::endl; \
  } while (0)

// A class that inherits from ::testing::Test and provides some common data and methods
class MyTest : public ::testing::Test {
 protected:
  // A method that runs before each test case
  void SetUp() override {
    // Initialize some data or objects here
    std::cout << "Setting up" << std::endl;
  }

  // A method that runs after each test case
  void TearDown() override {
    // Clean up some data or objects here
    std::cout << "Tearing down" << std::endl;
  }

  // Some common data or objects that can be used by each test case
  int a = 10;
  int b = 20;
};

// A test case that uses the MyTest class and tests some condition
TEST_F(MyTest, TestCondition) {
  // Use the common data or objects from the MyTest class
  std::cout << "a = " << a << ", b = " << b << std::endl;

  // Use some assertions to check the condition
  ASSERT_EQ(a + b, 30);
}

// Another test case that uses the MyTest class and tests another condition
TEST_F(MyTest, TestAnotherCondition) {
  // Use the common data or objects from the MyTest class
  std::cout << "a = " << a << ", b = " << b << std::endl;

  // Use some assertions to check the condition
  ASSERT_NE(a - b, 0);
}

// A main function that runs all the test cases
int main(int argc, char** argv) {
  // Initialize the testing framework
  ::testing::InitGoogleTest(&argc, argv);

  // Declare a boolean variable to indicate whether to loop or not
  bool loop = false;

  // Use some condition or command line argument to set the value of loop
  // For example, you can use a flag like --loop to enable looping
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--loop") == 0) {
      loop = true;
      break;
    }
  }

  // Check the value of loop and decide whether to execute the loop code or not
  if (loop) {
    // Loop over all the test cases randomly and print a message
    for (int i = 0; i < ::testing::UnitTest::GetInstance()->total_test_case_count(); i++) {
      // Generate a random index between 0 and total_test_case_count() -1 
      int index = rand() % ::testing::UnitTest::GetInstance()->total_test_case_count();
      // Get the test case at the random index 
      const ::testing::TestCase* test_case = ::testing::UnitTest::GetInstance()->GetTestCase(index);
      // Print a message before and after running the test case 
      PRINT_MSG(LOOP_100(RUN_ALL_TESTS()));
    }
  } else {
    // Just run all the test cases once without looping
    RUN_ALL_TESTS();
  }

}
