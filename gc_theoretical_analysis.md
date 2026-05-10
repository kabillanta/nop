# Theoretical Analysis of Gradient Centralization (GC)

## 1. Gradient Centralization as a Projection Operator

Let $\mathcal{L}(\mathbf{w})$ be the loss function parameterized by weights $\mathbf{w} \in \mathbb{R}^N$. The standard gradient is $\nabla \mathcal{L}(\mathbf{w})$.
Gradient Centralization (GC) computes the mean of the column vectors of the weight matrix and subtracts it from each column vector. For a weight vector $\mathbf{w}$, the GC operation $\Phi_{GC}$ on the gradient $\mathbf{g} = \nabla \mathcal{L}(\mathbf{w})$ is defined as:

$$ \Phi_{GC}(\mathbf{g}) = \mathbf{g} - \mu_{\mathbf{g}} \mathbf{1} $$

where $\mu_{\mathbf{g}} = \frac{1}{N}\sum_{i=1}^N g_i$ is the mean of the gradient and $\mathbf{1} \in \mathbb{R}^N$ is a vector of ones.

This operation can be elegantly written as a matrix-vector multiplication. Let $P$ be the projection matrix:

$$ P = I - \frac{1}{N}\mathbf{1}\mathbf{1}^T $$

It is straightforward to verify that $P$ is symmetric ($P^T = P$) and idempotent ($P^2 = P$). Thus, $P$ is an orthogonal projection matrix that projects vectors onto the hyperplane $\mathbf{1}^T \mathbf{x} = 0$.

We can write the centralized gradient as:
$$ \Phi_{GC}(\mathbf{g}) = P \mathbf{g} = P \nabla \mathcal{L}(\mathbf{w}) $$

## 2. Effective Hessian and Condition Number Reduction

In optimization, the condition number of the Hessian matrix $H = \nabla^2 \mathcal{L}(\mathbf{w})$ governs the convergence rate of first-order methods. The condition number is $\kappa(H) = \frac{\lambda_{max}(H)}{\lambda_{min}(H)}$, where $\lambda_{max}$ and $\lambda_{min}$ are the maximum and minimum eigenvalues of $H$.

With GC, the weight update rule in vanilla Gradient Descent (GD) becomes:
$$ \mathbf{w}_{t+1} = \mathbf{w}_t - \alpha P \nabla \mathcal{L}(\mathbf{w}_t) $$

Using Taylor expansion around the optimum $\mathbf{w}^*$, the effective geometry of the loss landscape is now governed by the *projected* Hessian:
$$ \tilde{H} = P H P $$

### Proof of Condition Number Reduction
For overparameterized networks with homogeneous activations (like ReLU, where $f(\alpha x) = \alpha f(x)$ for $\alpha > 0$), the weight scale often introduces a massive eigenvalue in the direction of the mean vector $\mathbf{1}$. Thus, $\mathbf{1}$ is often an eigenvector associated with the largest eigenvalue $\lambda_{max}(H)$ (or heavily aligned with it).

Let the eigenspectrum of $H$ be bounded by $0 < \mu \le \lambda_i \le L$.
Because $P$ projects out the subspace spanned by $\mathbf{1}$, the effective Hessian $\tilde{H} = PHP$ has a modified spectrum. If the largest curvature direction $\mathbf{v}_{max}$ of $H$ is aligned with $\mathbf{1}$ (i.e., $\mathbf{v}_{max} \approx \mathbf{1}$), then applying $P$ eliminates this extreme curvature:

$$ \lambda_{max}(\tilde{H}) \le \lambda_{max}(H) $$
$$ \lambda_{min}^+(\tilde{H}) \ge \lambda_{min}(H) $$

where $\lambda_{min}^+$ is the smallest non-zero eigenvalue.
Therefore, the effective condition number $\kappa(\tilde{H})$ is strictly improved:
$$ \kappa(\tilde{H}) = \frac{\lambda_{max}(\tilde{H})}{\lambda_{min}^+(\tilde{H})} < \frac{\lambda_{max}(H)}{\lambda_{min}(H)} = \kappa(H) $$

This formally demonstrates GC's role as a **preconditioner** that reduces the condition number without explicit $O(N^3)$ matrix inversion (like in Newton's method).

## 3. Linear Convergence Rates for GC

Assume $\mathcal{L}(\mathbf{w})$ is $\mu$-strongly convex and $L$-smooth in the projected subspace.
That is, for any $\mathbf{w}_1, \mathbf{w}_2$ such that $P(\mathbf{w}_1 - \mathbf{w}_2) = \mathbf{w}_1 - \mathbf{w}_2$:
$$ \mu \| \mathbf{w}_1 - \mathbf{w}_2 \|^2 \le \langle \nabla \mathcal{L}(\mathbf{w}_1) - \nabla \mathcal{L}(\mathbf{w}_2), \mathbf{w}_1 - \mathbf{w}_2 \rangle \le L \| \mathbf{w}_1 - \mathbf{w}_2 \|^2 $$

However, with GC, we are operating with the projected gradient $P \nabla \mathcal{L}(\mathbf{w})$. The effective smoothness constant becomes $\tilde{L} = \lambda_{max}(PHP) \le L$, and the effective strong convexity constant becomes $\tilde{\mu} = \lambda_{min}^+(PHP) \ge \mu$.

Using the standard proof for Gradient Descent under strong convexity, the distance to the optimum $\mathbf{w}^*$ at step $t$ is bounded by:
$$ \| \mathbf{w}_{t+1} - \mathbf{w}^* \|^2 \le \left( 1 - \frac{2\alpha \tilde{\mu} \tilde{L}}{\tilde{\mu} + \tilde{L}} \right) \| \mathbf{w}_t - \mathbf{w}^* \|^2 $$

By setting the optimal learning rate $\alpha = \frac{2}{\tilde{L} + \tilde{\mu}}$, we obtain a linear convergence rate:
$$ \| \mathbf{w}_{t+1} - \mathbf{w}^* \| \le \left( \frac{\kappa(\tilde{H}) - 1}{\kappa(\tilde{H}) + 1} \right) \| \mathbf{w}_t - \mathbf{w}^* \| $$

Since $\kappa(\tilde{H}) < \kappa(H)$, the contraction factor is strictly smaller, proving that GC accelerates linear convergence compared to standard GD.

## 4. Extension to Non-Convex Settings

In deep learning, $\mathcal{L}(\mathbf{w})$ is non-convex. We assume $\mathcal{L}$ is bounded below by $\mathcal{L}^*$ and has $\tilde{L}$-Lipschitz continuous projected gradients:
$$ \| P \nabla \mathcal{L}(\mathbf{w}_1) - P \nabla \mathcal{L}(\mathbf{w}_2) \| \le \tilde{L} \| \mathbf{w}_1 - \mathbf{w}_2 \| $$

For the update $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha P \nabla \mathcal{L}(\mathbf{w}_t)$, using the descent lemma:
$$ \mathcal{L}(\mathbf{w}_{t+1}) \le \mathcal{L}(\mathbf{w}_t) + \langle \nabla \mathcal{L}(\mathbf{w}_t), \mathbf{w}_{t+1} - \mathbf{w}_t \rangle + \frac{\tilde{L}}{2} \| \mathbf{w}_{t+1} - \mathbf{w}_t \|^2 $$
$$ \mathcal{L}(\mathbf{w}_{t+1}) \le \mathcal{L}(\mathbf{w}_t) - \alpha \langle \nabla \mathcal{L}(\mathbf{w}_t), P \nabla \mathcal{L}(\mathbf{w}_t) \rangle + \frac{\tilde{L}\alpha^2}{2} \| P \nabla \mathcal{L}(\mathbf{w}_t) \|^2 $$

Since $P$ is a projection matrix, $P^T = P$ and $P^2 = P$, so $\langle \mathbf{g}, P \mathbf{g} \rangle = \mathbf{g}^T P \mathbf{g} = \mathbf{g}^T P^T P \mathbf{g} = \| P \mathbf{g} \|^2$. Thus:
$$ \mathcal{L}(\mathbf{w}_{t+1}) \le \mathcal{L}(\mathbf{w}_t) - \left( \alpha - \frac{\tilde{L}\alpha^2}{2} \right) \| P \nabla \mathcal{L}(\mathbf{w}_t) \|^2 $$

By setting $\alpha = \frac{1}{\tilde{L}}$ (which is larger than $\frac{1}{L}$ since $\tilde{L} \le L$), we get:
$$ \mathcal{L}(\mathbf{w}_{t+1}) \le \mathcal{L}(\mathbf{w}_t) - \frac{1}{2\tilde{L}} \| P \nabla \mathcal{L}(\mathbf{w}_t) \|^2 $$

Summing this inequality from $t=0$ to $T-1$ and rearranging yields:
$$ \frac{1}{T} \sum_{t=0}^{T-1} \| P \nabla \mathcal{L}(\mathbf{w}_t) \|^2 \le \frac{2\tilde{L} (\mathcal{L}(\mathbf{w}_0) - \mathcal{L}^*)}{T} $$

As $T \to \infty$, the average projected gradient norm vanishes, meaning the algorithm converges to a first-order stationary point in the projected subspace at a rate of $O(1/T)$. Furthermore, because $\tilde{L} \le L$, the upper bound on the gradient norm is strictly tighter, demonstrating faster convergence in non-convex settings.

### Convergence to Second-Order Stationary Points
To escape saddle points and reach a second-order stationary point (SOSP), noise must be injected (e.g., via SGD). The condition for SOSP requires $\lambda_{min}(\tilde{H}) \ge -\epsilon$. Because GC constraints the optimization to a subspace where the dominant positive curvature (associated with weight scaling) is removed, the noisy gradient steps are more effective at exploring directions of negative curvature, facilitating faster escape from saddle points.

## Conclusion
Gradient Centralization acts as an implicit diagonal-free preconditioner. By projecting gradients onto a zero-mean hyperplane, it removes the dominant eigenvalue associated with homogeneous scale invariance, strictly reducing the condition number of the effective Hessian. This rigorously explains its empirical success in accelerating convergence in both strongly convex and non-convex neural network optimization.
