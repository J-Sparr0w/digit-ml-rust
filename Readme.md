# Digit Recognizer Neural Network 
* A neural network built using rust and ndarray lib for matrix manipulations. 
* Accuracy: approx. 85%

## Dataset
* from [kaggle](https://www.kaggle.com/competitions/digit-recognizer/data "digit recognizer dataset")
* each record has 785 columns
    *  column 1: label
    *  columns 2-785: pixel data ranging from 0-255
    

## Neural Network Architecture:
* Network has two layers
    * input => 785xm
    * output => 10xm
    * where m is the number of samples in training set

    * W(l) => weights of layer l {shape: 10x785}
    * b(l) => biases of layer l
    * A(l) => output after applying activation function on W(l) for Layer l
    * Z(l) => weighted sum of layer l (wx+b)
 
    * Forward Prop:
        * X = input (or A0)
        * Z1 = W1 . X + b1
        * A1 = ReLu(Z1)
        * Z2 = W2 . A1 + b2
        * A2 = Softmax(Z2)
        * output = A2
        * Flow:  X->Z1->A1->Z2->A2
 
    * Backward Prop:
        * dZ2 = A2 - Y
        * dW2 = (1/m) * (dZ2 . A1.T)
        * dB2 = (1/m) sum(dZ2)
        * dZ1 = W2.T.dZ2 * ReLU_deriv(Z1)
        * dW1 = (1/m) * (dZ1 . (X.T))
        * dB1 = (1/m) * sum(dZ1)
* Architecture and inspiration from: [@SamsonZhangTheSalmon](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=538s)

## How To Run
* Prerequisites: Install Rust
* Clone the repository: ```git clone https://github.com/J-Sparr0w/digit-ml-rust.git```
* Build the project: ```cargo run --release```

