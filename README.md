# ET-Autodiff

## A basic automatic differentiation library built with C++ expression templates.

 ```cpp
Et::ConstantExpr C1{ Et::Double(4) }, C2{ Et::Double(2) };
Et::VariableExpr X1{ Et::Double(5.53) }, X2{ Et::Double(-3.12) };
Et::PlaceholderExpr<Et::Double> P;
 ```

### Declaration of Constants,Variables & Placeholder objects.

```cpp
auto Y = X1 * X1 + X2 * X2 + C1 * X1 + C2 * X2 + P;
```

### Define the Cost function to minimize with respect to the Variables.
### y = f(x1,x2) = x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> + 4x<sub>1</sub> + 2x<sub>2</sub> - 6.3

```cpp
Et::GradientDescentOptimizer Optimizer{ Y };
```
### Create an Optimizer object which simplifies the training steps.

```cpp
int Iterations = 1000;
	for (int i = 0; i < Iterations; i++)
{
  std::cout << "Value at #" << i + 1 << " : " <<

  Optimizer
    .FeedPlaceholders(Et::PlFeed(P, -6.3))
    .Eval()
    .Backpass(0.01, X1, X2)
    .GetPreResult()
  
  << std::endl;
}
```

### Apply Backpropagation 1,000 times.

```cpp
std::cout << "Final Value :" << Y() << std::endl;
```

### Print the final value of the Cost function.
