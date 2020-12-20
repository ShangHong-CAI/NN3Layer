#ifndef NN_3_LAYER_H
#define NN_3_LAYER_H

#include <armadillo>

using namespace arma;

class NN3Layer {
    private:
    mat w;
    mat v;
    double learningRate;
    double sigmoid(double x){
        return  1 / (1 + exp(-x));
    }
    double sigmoid_de(double y){//Differential equation
        return  y * ( 1 - y );
    }
    mat mat_to_sigmoid(mat x){
      x.for_each( [&](mat::elem_type& val) { val=sigmoid(val); } );
      return x;
    }
    mat mat_to_sigmoid_de(mat x){
      x.for_each( [&](mat::elem_type& val) { val=sigmoid_de(val); } );
      return x;
    }
    mat mat_to_learning_rate(mat x){
      x.for_each( [&](mat::elem_type& val) { val=val*this->learningRate; } );
      return x;
    }
    public:
    NN3Layer(int inputLayer,int hiddenLayer,int outputLayer,double learningRate){
        this->learningRate = learningRate;
        this->w =mat(hiddenLayer,inputLayer+1,fill::randu);
        this->v =mat(outputLayer,hiddenLayer+1,fill::randu);
    };
    mat forward_input_to_hidden(mat x){
        mat h;
        //set bias
        mat set_bias(1, 1, fill::ones);
        x = join_rows(x,set_bias);
        //h=sigmoid(wx+bias)
        h=this->mat_to_sigmoid(this->w*x.t());
        return h;
    }
    mat forward_hidden_to_output(mat h){
        mat o;
        //set bias
        mat set_bias(1, 1, fill::ones);
        h = join_cols(h,set_bias);
        //o=sigmoid(vh+bias)
        o=this->mat_to_sigmoid(this->v*h);
        return o;
    }
    //dO
    mat backward_output_to_hidden(mat o,mat d){
      mat error = d-o;
      square(error).print("error:");
      //-1*(d-o)
      mat dError =error;
      dError.for_each( [&](mat::elem_type& val) { val=val*-1; } );
      //sigmoid_de
      return this->mat_to_sigmoid_de(o)%dError;
    }
    //dH
    mat backward_hidden_to_input(mat dO,mat h){
      mat dnet =this->mat_to_sigmoid_de(h);
      mat dH = this->v;
      //remove bias
      dH.reshape(this->v.n_rows,this->v.n_cols-1);
      return (dH.t()*dO) %dnet;
    }
    void backward_update(mat dO,mat h,mat dH,mat x){
      //set bias
      mat set_bias(1, 1, fill::ones);
      //calculate weight update value
      mat dV = dO * join_cols(h,set_bias).t();
      mat dW = dH * join_rows(x,set_bias);
      //update weight
      this->v = this->v - this->mat_to_learning_rate(dV);
      this->w = this->w - this->mat_to_learning_rate(dW);
    }
    void train_step_sgd(mat x,mat d){
      //forward
      mat h = this->forward_input_to_hidden(x);
      mat o = this->forward_hidden_to_output(h);
      //backward
      //calculate gradient
      mat dO = this->backward_output_to_hidden(o,d);
      mat dH = this->backward_hidden_to_input(dO,h);
      //update weight
      this->backward_update(dO,h,dH,x);
    }
    mat only_forward(mat x){
      //forward
      mat h = this->forward_input_to_hidden(x);
      return this->forward_hidden_to_output(h);
    }



};

#endif