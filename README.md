# Volatility Smile Optimization via SVI Model Utilizing NLP Algorithms

## 0. Special Thanks
This study was inspired by research conducted based on _'A New Simple Approach for Constructing Implied Volatility Surfaces'_ by Peter Carr and Liuren Wu.

## I. Research Background and Objectives

### 1.1 Background
**1.1.1 1987 Black Monday:** Highlighted the risks of extreme volatility, which can cause sudden market shocks and lead to unfair option prices and arbitrage opportunity.
**1.1.2 Recent Volatility Shock (August 5, 2024)**: Global markets, especially tech stocks, experienced sharp declines, with the VIX surging over 65%.  
**1.1.3 Limitations of Black-Scholes Model**: Assumes constant volatility, failing to account for the asymmetric and extreme volatility patterns observed in real markets, leading to option mispricing and arbitrage opportunities.  
**1.1.4 Introduction of SVI Model**: The Stochastic Volatility Inspired (SVI) model addresses some of these limitations by dynamically modeling implied volatility and explaining the volatility smile in extreme market conditions.  
**1.1.5 Need for SVI Model Improvement**: Even the SVI model has limitations in fully capturing volatility asymmetry. Hence, this study proposes a Post-SVI model incorporating "Convex Duality" to improve option pricing and dynamic hedging in extreme volatility scenarios.

### 1.2 Research Objectives
**1.2.1 Improve Option Pricing Accuracy**: The goal is to enhance option pricing during extreme volatility conditions using the Post-SVI model, incorporating "Convex Duality" to better estimate implied volatility.  
**1.2.2 Eliminate Arbitrage Opportunities**: By improving the modeling of volatility smiles and addressing mispricing, the study aims to create an arbitrage-free environment.  
**1.2.3 Enhance Dynamic Delta Hedging**: Utilize real-time dynamic delta hedging strategies based on the refined Post-SVI model to improve risk management and market stability during volatile periods.

## II. Literature Review
### 2.1 Arbitrage-Free SVI Volatility Surfaces
**2.1.1 SVI and Static Arbitrage**: Jim Gatheral and Antoine Jacquier's work on "Arbitrage-Free SVI Volatility Surfaces" demonstrates how SVI can be used to prevent static arbitrage in options pricing. The SVI model ensures no arbitrage opportunities are arised through fitting the model well with market data.

![image](https://github.com/user-attachments/assets/ee853630-eb1c-4c2e-90eb-ce51379f381c)

**2.1.2 Types of Static Arbitrage:**
- **Calendar Spread Arbitrage**: This occurs when options with different expirations on the same underlying asset violate price symmetry. The model ensures that volatility surfaces increase monotonically over time to avoid arbitrage.
- **Butterfly Arbitrage**: This happens due to irregular curvature in implied volatility across strike prices. The SVI model controls the curvature, preventing extreme volatility spikes or drops that could lead to risk-free profits.

**2.1.3 Lee's Formula and Volatility Asymmetry:**
Lee's formula limits the growth of implied volatility at extreme strike prices to avoid unrealistic volatility smiles. The SVI model incorporates this through its parameters *(a, b, ρ, m, σ)*, ensuring a balanced and realistic volatility surface.

### 2.2 SVI Parameter Calibration

**2.2.1 Parameter Importance:**
The initial parameter setup in the SVI model significantly influences its ability to fit the volatility surface and prevent arbitrage. The parameters *a, b, ρ, m,* and *σ* must be properly constrained to ensure realistic and stable volatility surface formation.

![image](https://github.com/user-attachments/assets/368fd3a0-996d-40ea-b3ff-7f542e3f2089)

**2.2.2 Parameter Range:**

$$
0 \leq a \leq \max_{i} [\omega_i],
$$

$$
0 \leq b \leq \frac{4}{\tau (1 + |\rho|)},
$$

$$
-1 \leq \rho \leq 1,
$$

$$
\min_{i}[x_i] \leq m \leq \max_{i}[x_i],
$$

$$
0 < \sigma_{\min} \leq \sigma \leq 10.
$$

These constraints help ensure that the SVI model's parameters *(a, b, ρ, m, σ)* are set within reasonable and realistic ranges, allowing the model to avoid arbitrage while accurately fitting the implied volatility surface.

## III. Stochastic Volatility Inspired (SVI) Model

### 3.1 SVI Model

$$
w(k) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)
$$

The Stochastic Volatility Inspired (SVI) model was introduced by Jim Gatheral of Merrill Lynch in 1999. It is a simple parametric model used to explain the volatility smile observed in the options market. The SVI model expresses implied volatility as a function of the option’s time to maturity and the difference between the strike price and the underlying asset price.

**1. Key Concept**:  
   The primary purpose of the SVI model is to fit the volatility smile with a smooth and convex function, allowing it to accurately capture the implied volatility of options across different strike prices and maturities. A convex volatility curve ensures that implied volatility behaves consistently and avoids theoretical mispricing, leading to more accurate option pricing.

**2. Convexity and Arbitrage**:  
   Convexity plays a crucial role in ensuring that the volatility smile is stable and rational across the market. A non-convex or irregular volatility smile could indicate arbitrage opportunities where option prices deviate from theoretical values. By using the SVI model to fit a convex curve, such arbitrage opportunities are minimized, improving market efficiency.

**3. Risk Management**:  
   The convexity of the volatility smile is also vital for managing risk and developing hedging strategies. When volatility is expressed smoothly and consistently, key Greeks like delta, gamma, and vega can be calculated more accurately, allowing for more effective portfolio risk management and more precise hedging strategies, such as delta or gamma hedging.

### 3.2 SVI Jump-Wings Parameterization
The **SVI Jump-Wings Parameterization** is inspired by the approach proposed by Tim Klassen, designed to reflect changes in the implied volatility surface more realistically. This parameterization uses implied variance instead of total implied variance and explicitly incorporates time-to-expiration dependency. It is defined using the following raw parameters:

$$
\nu_\tau = \frac{a + b \left( - \rho m + \sqrt{m^2 + \sigma^2} \right)}{\tau},
$$

$$
\phi_\tau = \frac{1}{\sqrt{\omega_\tau}} \frac{b}{2} \left( \rho - \frac{m}{\sqrt{m^2 + \sigma^2}} \right),
$$

$$
p_\tau = \frac{1}{\sqrt{\omega_\tau}} b (1 - \rho),
$$

$$
c_\tau = \frac{1}{\sqrt{\omega_\tau}} b (1 + \rho),
$$

$$
\hat{\nu}_\tau = \frac{1}{\tau} \left( a + b \sigma \sqrt{1 - \rho^2} \right).
$$

### 3.3 SVI Model Optimization via Non-Linear Programming (NLP)

![image](https://github.com/user-attachments/assets/8423b253-ff59-4cc6-a616-d9a2b0cec61f)


The goal is to optimize this volatility smile using the SVI model. For details on data acquisition and preprocessing, please refer to the 'Data` folder.

**3.3.1 Objective Function in the SVI Model**
The objective function, along with constraints, is classified as a non-linear programming (NLP) problem. In the SVI model, the function that represents the volatility smile is defined as follows:

$$
w(k) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)
$$

This function models the implied volatility curve based on several parameters to fit the volatility smile. It indicates how volatility changes with the strike price, exhibiting a non-linear relationship. Therefore, optimizing the SVI model involves solving a non-linear optimization problem to ensure that the volatility smile is properly fitted.

## IV. Project Methodology

### 4.1 Definition of the Primal Problem
The primal objective is to minimize the error between the market-observed total variance and the total variance calculated by the SVI model. The Mean Squared Error (MSE) or Mean Absolute Error (MAE) is typically used to achieve this, and the primal problem is structured as:

$$
\min_{a, b, \rho, m, \sigma} \sum_{i=1}^{N} \left( \hat{w}(k_i) - w(k_i) \right)^2
$$

Where N represents the number of volatility data points. The constraints ensuring convexity and arbitrage-free conditions are as follows:

$$
b > 0, \quad -1 < \rho < 1, \quad \sigma > 0
$$

### 4.2 Lagrangian Transformation
The primal problem is converted into a Lagrangian to account for the imposed constraints. The Lagrangian is given by:

$$
L(a, b, \rho, m, \sigma, \lambda) = \sum_{i=1}^{N} \left( \hat{w}(k_i) - w(k_i) \right)^2 + \lambda_1 (b - 0) + \lambda_2 (\rho (1 - 1)) + \lambda_3 (\sigma - 0)
$$

Here, lambda1, lambda2, and lambda3 are the Lagrange multipliers that enforce the constraints.

### 4.3 Types of NLP Optimization Algorithms
- **Gradient-based Algorithms**:  
  The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm computes gradients and Hessians to achieve fast convergence, making it suitable for large-scale non-linear problems. L-BFGS (Limited-memory BFGS) is a variation of BFGS that reduces memory usage and is more efficient for large datasets.

- **Stochastic Optimization Algorithms**:  
  Simulated Annealing is designed to search for the global optimum of the objective function by exploring multiple directions from an initial solution. The Genetic Algorithm applies biological principles such as crossover and mutation to search the parameter space for the global optimum.

### 4.4 Avoiding and Improving Local Optima
A common issue in non-linear optimization is getting stuck in a **local optimum**, which represents the minimum value of the objective function within a specific parameter range but is not the global minimum across the entire parameter space. The goal of optimization is to find the **global optimum**, which is the minimum value of the objective function over the entire parameter space. Achieving convexity is crucial to ensuring that the global optimum is reached. The following methods can help avoid and improve the handling of local optima:

![image](https://github.com/user-attachments/assets/ad61581c-4d0c-4e95-9be1-b889f1b587d4)

### 4.5 Optimization using Convex Duality

4.5.1 Definition of the Dual Problem
By applying convex duality to the objective function and constraints presented in section 3.4.5.2, the primal problem can be transformed into a dual problem. This transformation can reduce the complexity of optimization or provide a new interpretation, allowing for the development of more efficient algorithms. The dual optimization problem for the SVI model is formulated as:

$$
\max_{\lambda} \; \inf_{a, b, \rho, m, \sigma} \; L(a, b, \rho, m, \sigma, \lambda)
$$


This equation redefines the Lagrangian function, where the primal variables *(a), (b), (ρ), (m), (σ)* are minimized, and the Lagrange multiplier *(λ)* is maximized. Compared to the primal problem, the dual problem has relaxed constraints, reducing the parameter space, making the optimization more efficient, and facilitating easier access to the global optimum. Furthermore, instead of applying penalties for violating constraints in the primal problem, the dual problem directly handles constraints through Lagrange multipliers, simplifying the optimization process.

**4.5.2 Strong Duality and the Dual Gap**
Strong duality occurs when the convexity condition is satisfied, meaning the optimal solutions of the primal and dual problems coincide. In other words, the optimal value of the objective function for both the primal and dual problems will be equal. When strong duality holds, the **dual gap** is zero. The dual gap is the difference between the optimal objective function values of the primal and dual problems. By verifying the duality gap in the redefined objective function in section 3.4.6.1, the optimization can be confirmed as successful when the primal and dual solutions match.

## V. Optimization

### 5.1 Gradient Descent
The first optimization method used was gradient descent. Gradient descent is an optimization algorithm that adjusts parameter values in the direction where the objective function decreases, guiding the function toward a global optimum. However, this method has limitations such as the likelihood of getting stuck in a local optimum based on the chosen initial values and the challenge of setting an appropriate step size. After optimizing with the following settings, further optimization was performed by adjusting the settings:

- **Learning rate**: 0.001  
- **Iterations**: 5000  
- **Tolerance**: 1e-6  
- **Epsilon for numerical stability**: 1e-8  

After classifying the option trading data by expiration dates, gradient descent was applied to optimize the volatility smile. The results are shown below:

![image](https://github.com/user-attachments/assets/6a7372b8-67e0-4529-abb3-b5df38ad4073)


**Key Findings:**
1. The volatility smile could not be derived in some cases.
2. The derived volatility smile did not meet convexity conditions.
3. Proper fitting (regression) was not achieved.

After running 5000 iterations again, the optimization was re-evaluated.

![image](https://github.com/user-attachments/assets/573ce1ab-7527-4f5f-ab51-59c47a759aa3)


### 5.2 L-BFGS-B
The L-BFGS-B algorithm was effective for the following reasons:

- **Limited-memory**: Minimizes memory usage while handling large-scale problems. Effective when the parameter space is large or there are many data points.
- **BFGS**: Uses approximations rather than calculating second-order derivatives directly, converging more efficiently than general gradient descent algorithms.
- **Box Constraints**: Directly controls variable ranges, preventing improper values from being generated during the optimization process. Boundary conditions can be applied to optimization variables.

Given that the data being used is very large, and the number of parameters being searched is five (in a 5-dimensional space), the L-BFGS-B algorithm was deemed appropriate for optimization.

![image](https://github.com/user-attachments/assets/a79d7c32-b39d-4985-863c-535bae1068a6)
![image](https://github.com/user-attachments/assets/4842c1f6-f8d6-4320-a7d4-b2815cf65dd4)

The blue dots represent the actual volatility for the trading data. The same arrangement was observed in both algorithms. Considering this, it can be visually confirmed that the L-BFGS-B algorithm fits the data points much better than gradient descent.

![image](https://github.com/user-attachments/assets/dd94015b-a865-4a89-baed-c943e6ae7b4e)

**Key Findings:**
1. The L-BFGS-B algorithm performed better than gradient descent (visually noticeable).
2. For some expiration dates (Expiration: 2024-08-16), overfitting appeared to be applied excessively.

**Further Optimization Conducted:**
- Applying convexity constraints for all log strikes : Instead of using all possible values of (k) to check convexity, sample points of (k) were generated at regular intervals to verify convexity.
  
  ![image](https://github.com/user-attachments/assets/efa30503-225f-4cd9-a890-5e7fa88c91fd)

- Smoothing the curve : To prevent overfitting in the SVI model, a smoothing term was added to the objective function, reducing model variance and strengthening convexity.
  
  ![image](https://github.com/user-attachments/assets/eed87122-a286-4a20-9c93-92570bf9ea0f)

- Setting a lower bound for second derivatives : The first two methods did not work out well, so this method was implemented
  
  ![image](https://github.com/user-attachments/assets/24791d8d-4774-408b-8f15-6ce26a50d36d)

**Additional Optimization Purpose:**
1. Ensuring the curve takes on a fully convex shape by setting the second derivative not just greater than zero, but above a small value ( \epsilon ).
2. Ensuring the curve does not become too flat and consistently remains convex.

**Analysis:**
1. The convexity of the volatility smile was generally satisfied.
2. Some parts were still not properly fitted.
3. It appeared to be better optimized compared to other methods.
Overall, the fitting results showed a convex shape, excluding some expiration date option trading data.

## VI. Result

### 6.1 Parameter
The following are the optimized parameter values for each expiration date derived from the L-BFGS-B optimization:

![image](https://github.com/user-attachments/assets/b4393d14-2cf7-4258-817c-d79be13f1d1b)


### 6.2 Performance Analysis: RMSE
RMSE (Root Mean Squared Error) is a metric that measures the difference between the predicted model and the actual values. A lower RMSE indicates that the model's predictions are closer to the actual values, implying better optimization performance.

![image](https://github.com/user-attachments/assets/5619bdce-532c-4743-bfe4-e228bf7607c7)

**Analysis 1: High RMSE on August 5**
The data used in this study consists of option trades made on August 5. The market volatility for this expiration date was much higher than for other dates, which may result in a higher loss value for the data. Additionally, with a higher trade volume, more data points are included, which can further increase the observed loss for this date.

**Analysis 2: Overall Low RMSE**
Despite the higher volatility on August 5, the RMSE is approximately 0.24, and the RMSE for option trades with other expiration dates is significantly lower. This demonstrates that the overall optimization was successful.

### 6.3 Performance Analysis: Objective Function Value
By iteratively optimizing, we can analyze the performance by checking how much the initial objective function value has decreased. The larger the decrease, the better the optimization performance.

![image](https://github.com/user-attachments/assets/9ad2d0b3-c037-40ca-8103-e58a745f7e91)

**Analysis 1: Orange Line Below Blue Line**
The blue dashed line represents the initial objective function value, and the orange solid line represents the objective function value after iterative optimization. For notable expiration date data (August 5 and August 16), the post-optimization objective function value is significantly lower than the pre-optimization value.

**Analysis 2: What about Other Dates?**
Apart from the prominent expiration dates, the changes in objective function values for other expiration dates are not easily discernible. We visualize the data again, excluding the data for these two expiration dates.

![image](https://github.com/user-attachments/assets/14dc7063-6b32-48a7-8fbb-ba7f6d74b0a1)

After excluding the notable expiration dates (2024-08-05, 2024-08-16), we examined the decrease in the objective function values for the remaining expiration date option trade data. The figure compares the pre- and post-optimization objective function values.

Below is a visualization that highlights only the decrease in the objective function values.

![image](https://github.com/user-attachments/assets/210779d5-4242-41cd-b9af-eaf5843d8fc5)

The optimization process clearly reduced the objective function values. First, the absence of negative decreases indicates that no data points experienced an increase in the objective function value (i.e., no reverse optimization effect). Furthermore, even for option trades with objective function values close to zero, we can observe a slight decrease in the values.

### 6.4 Butterfly Arbitrage-Free Condition
We check whether the optimization of the volatility smile resulted in an arbitrage-free scenario for options. This study verifies the presence or absence of butterfly arbitrage as discussed in the literature review.

![image](https://github.com/user-attachments/assets/eba2e0a5-4826-4053-9aeb-96e2b251c2ab)
![image](https://github.com/user-attachments/assets/f3171179-14b3-4ad4-aa08-debc242507af)

**Analysis 1: Intuitive Analysis**
In the end, no butterfly arbitrage was observed in any of the expiration date option data analyzed. This indicates that the volatility smile, as estimated by the SVI model, was convexly optimized for all log strike prices.

**Analysis 2: Mathematical Analysis**
Considering the SVI model's equation, this means that the second derivative of w(k) is positive for all values of k. In other words, the differentiability shows that the volatility smile is continuous and smooth across all data points, without any sharp peaks, and the positive derivative confirms that the smile is convex.




