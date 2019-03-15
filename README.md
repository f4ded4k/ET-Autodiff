# ET-Autodiff

## A basic automatic differentiation library built with C++ expression templates.

 ```cpp
Et::ConstantExpr C1 = Et::Double(4), C2 = Et::Double(2);
Et::VariableExpr X1 = Et::Double(5.53), X2 = Et::Double(-3.12);
Et::PlaceholderExpr<Et::Double> P;
 ```

### Declaration of Constants,Variables & Placeholder objects.

```cpp
auto Y = X1 * X1 + X2 * X2 + C1 * X1 + C2 * X2 + P;
```

### Define the Cost function to minimize with respect to the Variables.

```cpp
int Iterations = 1000;
for (int i = 0; i < Iterations; i++)
{
  std::cout << "Value at #" << i + 1 << " : " << Et::Evaluate(Y, Et::PlFeed(P, -6.3)) << std::endl;
  Et::ApplyGrediants(0.01, X1, X2);
}
```

### Apply Backpropagation 1,000 times.

```cpp
std::cout << "Final Value :" << Y() << std::endl;
```

### Print the final value of the Cost function.
