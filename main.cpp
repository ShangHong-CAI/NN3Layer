#include <iostream>
#include <armadillo>
#include <random>
#include <math.h>
#include "NN3Layer.hpp"
using namespace std;
using namespace arma;

mat gen_and_example(int randNum){
    mat A(1,3);
    switch (randNum)
    {
    case 0:
        A={{0,0,0}};
        break;
    case 1:
        A={{0,1,0}};
        break;
    case 2:
        A={{1,0,0}};
        break;
    case 3:
        A={{1,1,1}};
        break;
    }
    return A;
}
mat gen_or_example(int randNum){
    mat A(1,3);
    switch (randNum)
    {
    case 0:
        A={{0,0,0}};
        break;
    case 1:
        A={{0,1,1}};
        break;
    case 2:
        A={{1,0,1}};
        break;
    case 3:
        A={{1,1,1}};
        break;
    }
    return A;
}
mat gen_xor_example(int randNum){
    mat A(1,3);
    switch (randNum)
    {
    case 0:
        A={{0,0,0}};
        break;
    case 1:
        A={{0,1,1}};
        break;
    case 2:
        A={{1,0,1}};
        break;
    case 3:
        A={{1,1,0}};
        break;
    }
    return A;
}


int main(int argc, char** argv){
      /* 隨機設備 */
  std::random_device rd;

  /* 亂數產生器 梅森旋轉演算法*/
  std::mt19937 generator( rd() );

  /* 亂數的機率分布 */
  std::uniform_real_distribution<float> unif(1.0, 4.0);
  NN3Layer nn3Layer(2,2,1,0.3);
  //train
  for( int a = 0; a < 10000; a++){
      /* 產生亂數 */
      float x = unif(generator);
      mat data = gen_xor_example((int)x);
      nn3Layer.train_step_sgd({data(0,0),data(0,1)},{data(0,2)});
  }
  //model use
  mat testData = gen_xor_example((int)unif(generator));
  testData.print("testData:");
  mat o = nn3Layer.only_forward({testData(0,0),testData(0,1)});
  o.print("output:");

}