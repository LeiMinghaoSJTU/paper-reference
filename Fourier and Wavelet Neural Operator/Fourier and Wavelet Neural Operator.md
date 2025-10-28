## **2.1 设定**

我们固定一个空间维度 $ d \in \mathbb{N} $，并用 $ D \subset \mathbb{R}^d $ 表示 $ \mathbb{R}^d $ 中的一个区域。
我们考虑算子 $ \mathcal{G}: \mathcal{A}(D; \mathbb{R}^{d_a}) \to \mathcal{U}(D; \mathbb{R}^{d_u}) $ 的近似，即 $ a \mapsto u := \mathcal{G}(a) $，其中：
输入 $ a \in \mathcal{A}(D; \mathbb{R}^{d_a}) $，$ d_a \in \mathbb{N} $，是一个从 $ D \to \mathbb{R}^{d_a} $ 的函数，具有 $ d_a $ 个分量；
输出 $ u \in \mathcal{U}(D; \mathbb{R}^{d_u}) $，$ d_u \in \mathbb{N} $，是一个从 $ D \to \mathbb{R}^{d_u} $ 的函数，具有 $ d_u $ 个分量。
这里 $ \mathcal{A}(D; \mathbb{R}^{d_a}) $ 和 $ \mathcal{U}(D; \mathbb{R}^{d_u}) $ 是巴拿赫空间（或巴拿赫空间的适当子集）。$ \mathcal{A} $ 和 $ \mathcal{U} $ 的典型例子包括连续函数空间 $ C(D; \mathbb{R}^{d_u}) $，或阶数 $ s \geq 0 $ 的索伯列夫空间 $ H^s(D; \mathbb{R}^{d_u}) $（详见附录 B 的定义）。

涉及偏微分方程（PDE）解算子的具体算子 $ \mathcal{G} $ 的例子见第 3 节。

## **2.2. 神经算子定义**

在上述设置 2.1 的基础上，并如文献 [18] 所定义，一个神经算子 $ \mathcal{N}: \mathcal{A}(D; \mathbb{R}^{d_a}) \to \mathcal{U}(D; \mathbb{R}^{d_u}) $，即 $ a \mapsto \mathcal{N}(a) $，具有如下形式：

$$
\mathcal{N}(a) = \mathcal{Q} \circ \mathcal{L}_L \circ \mathcal{L}_{L-1} \circ \cdots \circ \mathcal{L}_1 \circ \mathcal{R}(a),
$$

对于给定的深度 $ L \in \mathbb{N} $，其中 $ \mathcal{R}: \mathcal{A}(D; \mathbb{R}^{d_a}) \to \mathcal{U}(D; \mathbb{R}^{d_v}) $，$ d_v \geq d_u $，是一个提升算子（局部作用），在实践中，$ \mathcal{R} $ 通常是一个前馈神经网络（FNN）。我们这里假设其形式为
$$
\mathcal{R}(a)(x) = \sigma_{\mathcal{R}}\left( W_{\mathcal{R}} a(x) + b_{\mathcal{R}}(x) \right), \quad \forall x \in D. \tag{2.1}
$$

这里我们直接让激活函数 $\sigma_{\mathcal{R}}(x)=x$，且 $b_{\mathcal{R}}(x)=0$。

且 $ \mathcal{Q}: \mathcal{U}(D; \mathbb{R}^{d_v}) \to \mathcal{U}(D; \mathbb{R}^{d_u}) $ 是一个局部投影算子，通常也是一个 FNN。


与经典的有限维神经网络类似，层 $ \mathcal{L}_1, \ldots, \mathcal{L}_L $ 是非线性算子层，$ \mathcal{L}_\ell: \mathcal{U}(D; \mathbb{R}^{d_v}) \to \mathcal{U}(D; \mathbb{R}^{d_v}) $，$ v \mapsto \mathcal{L}_\ell(v) $，我们假设其形式为

$$
\mathcal{L}_\ell(v)(x) = \sigma\left( W_\ell v(x) + b_\ell(x) + (\mathcal{K}(a; \theta_\ell)v)(x) \right), \quad \forall x \in D.
$$

这里，权重矩阵 $ W_\ell \in \mathbb{R}^{d_v \times d_v} $ 和偏置 $ b_\ell(x) \in \mathcal{U}(D; \mathbb{R}^{d_v}) $ 定义了一个仿射逐点映射 $ W_\ell v(x) + b_\ell(x) $。在无限维设置中，线性算子的丰富性可以通过定义以下**非局部**线性算子来部分实现：

$$
\mathcal{K}: \mathcal{A} \times \Theta \to L\left( \mathcal{U}(D; \mathbb{R}^{d_v}), \mathcal{U}(D; \mathbb{R}^{d_v}) \right),
$$

该算子将输入场 $ a $ 和参数集 $ \Theta $ 中的参数 $ \theta \in \Theta $ 映射到一个有界线性算子 $ \mathcal{K}(a, \theta): \mathcal{U}(D; \mathbb{R}^{d_v}) \to \mathcal{U}(D; \mathbb{R}^{d_v}) $，而非线性激活函数 $ \sigma: \mathbb{R} \to \mathbb{R} $ 逐分量应用。如文献 [18] 所提出，线性算子 $ \mathcal{K}(a, \theta) $ 是如下形式的积分算子：

$$
(\mathcal{K}(a; \theta)v)(x) = \int_D \kappa_\theta(x, y; a(x), a(y)) v(y)\, dy, \quad \forall x \in D. \tag{2.3}
$$

这里，积分核 $ \kappa_\theta: \mathbb{R}^{2(d + d_a)} \to \mathbb{R}^{d_v \times d_v} $ 是由 $ \theta \in \Theta $ 参数化的神经网络。积分核 (2.3) 的具体例子包括在文献 [18] 中使用图核网络评估的核，或在文献 [17] 中使用多极展开的核。

## **2.3. 傅里叶神经算子定义**

如文献 [19] 所定义，傅里叶神经算子（FNOs）是通用神经算子 (2.3) 的特例，其中核 $ \kappa_\theta(x, y; a(x), a(y)) $ 具有形式 $ \kappa_\theta = \kappa_\theta(x - y) $。在这种情况下，(2.3) 可以写成卷积形式：

$$
(\mathcal{K}(\theta)v)(x) = \int_D \kappa_\theta(x - y) v(y)\, dy, \quad \forall x \in D. \tag{2.4}
$$

为了具体起见，我们考虑周期域 $ D = \mathbb{T}^d $（我们将其等同于标准环面 $ \mathbb{T}^d = [0, 2\pi]^d $），尽管非周期矩形域 $ D $ 也可以以直接方式处理。


在这一周期性框架下，公式 (2.4) 中的卷积算子可以通过傅里叶变换 $ \mathcal{F} $ 和逆傅里叶变换 $ \mathcal{F}^{-1} $ 来计算（参见附录 B 中的 (B.1) 和 (B.2) 以获取符号和定义），从而得到核 (2.3) 的如下等价表示：

$$
(\mathcal{K}(\theta)v)(x) = \mathcal{F}^{-1}\left( P_\theta(k) \cdot \mathcal{F}(v)(k) \right)(x), \quad \forall x \in \mathbb{T}^d. \tag{2.5}
$$

这里，$ P_\theta(k) \in \mathbb{C}^{d_v \times d_v} $ 是一个由 $ k \in \mathbb{Z}^d $ 索引的满矩阵，它通过傅里叶变换与公式 (2.4) 中的积分核 $ \kappa_\theta(x) $ 相关联，即 $ P_\theta(k) = \mathcal{F}(\kappa_\theta)(k) $。需要注意的是，我们必须施加条件 $ P_\theta(-k) = P_\theta(k)^\dagger $，即对所有 $ k \in \mathbb{Z}^d $，其值等于厄米转置，以确保当输入函数 $ v(x) $ 为实值时，输出函数 $ (\mathcal{K}(\theta)v)(x) $ 也是实值函数。
【原因】 因为傅里叶变换将实值函数映射到满足 $ \hat{v}(-k) = \hat{v}(k)^\dagger $ 的复值函数。


因此，对于周期域 $ \mathbb{T}^d $，傅里叶神经算子（FNOs）的形式是一个映射 $ \mathcal{N}: \mathcal{A}(D; \mathbb{R}^{d_a}) \to \mathcal{U}(D; \mathbb{R}^{d_u}) $，具体形式为

$$
\mathcal{N}(a) := \mathcal{Q} \circ \mathcal{L}_L \circ \mathcal{L}_{L-1} \circ \cdots \circ \mathcal{L}_1 \circ \mathcal{R}(a), \tag{2.6}
$$

其中提升算子 $ \mathcal{R} $ 和投影算子 $ \mathcal{Q} $ 分别由公式 (2.1) 和 (2.2) 给出，而非线性层 $ \mathcal{L}_\ell $ 的形式为

$$
\mathcal{L}_\ell(v)(x) = \sigma\left( W_\ell v(x) + b_\ell(x) + \mathcal{F}^{-1}\left( P_\ell(k) \cdot \mathcal{F}(v)(k) \right)(x) \right). \tag{2.7}
$$

这里，$ W_\ell \in \mathbb{R}^{d_v \times d_v} $ 和 $ b_\ell(x) $ 定义了一个逐点仿射映射（对应于权重和偏置），而 $ P_\ell: \mathbb{Z}^d \to \mathbb{C}^{d_v \times d_v} $ 通过傅里叶变换定义了一个非局部线性映射的系数。

对于本工作的其余部分，我们做出以下假设，

**假设 2.4**（激活函数）。除非另有明确说明，公式 (2.7) 中的激活函数 $\sigma: \mathbb{R} \to \mathbb{R}$ 被假设为非多项式、（全局）利普希茨连续且 $\sigma \in C^3$。

## 2.4. **FNO 的通用逼近性**
接下来，我们将证明 FNOs (2.6) 是“通用”的，即：给定在设定 2.1 中定义的一类算子，可以找到一个 FNO 来以任意所需的精度逼近它。为了更精确，我们有如下定理。

**定理 2.5**（通用逼近性）。设 $s, s' \geq 0$。设 $\mathcal{G}: H^s(\mathbb{T}^d; \mathbb{R}^{d_a}) \to H^{s'}(\mathbb{T}^d; \mathbb{R}^{d_u})$ 是一个连续算子。设 $K \subset H^s(\mathbb{T}^d; \mathbb{R}^{d_a})$ 是一个紧子集。那么对于任意 $\varepsilon > 0$，存在一个形式为 (2.6) 的 FNO $\mathcal{N}: H^s(\mathbb{T}^d; \mathbb{R}^{d_a}) \to H^{s'}(\mathbb{T}^d; \mathbb{R}^{d_u})$，作为算子 $H^s \to H^{s'}$ 连续，使得

$$
\sup_{a \in K} \|\mathcal{G}(a) - \mathcal{N}(a)\|_{H^{s'}} \leq \varepsilon.
$$

### **证明概要**

该通用逼近定理的详细证明见附录 D.4，此处我们简要概述。为简化记号，我们设 $d_a = d_u = 1$，并首先观察附录 D.1 中证明的如下引理：

**引理 2.6**。假设通用逼近定理 2.5 在 $s' = 0$ 时成立。那么它对任意 $s' \geq 0$ 也成立。

因此，主要目标是证明定理 2.5 在特殊情形 $s' = 0$ 下成立；即：给定一个连续算子 $\mathcal{G}: H^s(\mathbb{T}^d) \to L^2(\mathbb{T}^d)$，紧集 $K \subset H^s(\mathbb{T}^d)$，以及 $\varepsilon > 0$，我们希望构造一个 FNO $\mathcal{N}: H^s(\mathbb{T}^d) \to L^2(\mathbb{T}^d)$，使得

$$
\sup_{a \in K} \|\mathcal{G}(a) - \mathcal{N}(a)\|_{L^2} \leq \varepsilon.
$$

为此，我们首先定义如下算子：

$$
\mathcal{G}_N: H^s(\mathbb{T}^d) \to L^2(\mathbb{T}^d), \quad \mathcal{G}_N(a) := P_N \mathcal{G}(P_N a), \tag{2.8}
$$

其中 $P_N$ 是正交傅里叶投影算子，其定义为：

$$
P_N a(x) := \sum_{|k|_\infty \leq N} \hat{a}_k e^{ik \cdot x}, \quad \forall x \in \mathbb{T}^d.
$$

因此，$\mathcal{G}_N$ 可以粗略地看作是连续算子 $\mathcal{G}$ 的傅里叶投影。

接下来，我们可以证明：对于任意给定的 $\varepsilon > 0$，存在 $N \in \mathbb{N}$，使得

$$
\|\mathcal{G}(a) - \mathcal{G}_N(a)\|_{L^2} \leq \varepsilon, \quad \forall a \in K. \tag{2.9}
$$

因此，证明的关键在于找到一个 FNO (2.6)，使其能够以任意所需的精度逼近算子 $\mathcal{G}_N$。

为此，我们引入一组傅里叶波数 $k \in \mathcal{K}_N$，定义为

$$
\mathcal{K}_N := \left\{ k \in \mathbb{Z}^d \mid |k|_\infty \leq N \right\}, \tag{2.10}
$$

并定义一个形式为 $\hat{\mathcal{G}}_N: \mathbb{C}^{\mathcal{K}_N} \to \mathbb{C}^{\mathcal{K}_N}$ 的傅里叶共轭或傅里叶对偶算子，

$$
\hat{\mathcal{G}}_N(\hat{a}_k) := \mathcal{F}_N \left( \mathcal{G}_N \left( \mathrm{Re} \left( \mathcal{F}_N^{-1}(\hat{a}_k) \right) \right) \right), \tag{2.11}
$$

使得恒等式（注意 $P_N a$ 是实值函数，所以从2.11可以推出2.12）

$$
\mathcal{G}_N(a) = \mathcal{F}_N^{-1} \circ \hat{\mathcal{G}}_N \circ \mathcal{F}_N(P_N a), \tag{2.12}
$$

对所有实值 $a \in L^2(\mathbb{T}^d)$ 成立。这里，$\mathcal{F}_N$ 是离散傅里叶变换，而 $\mathcal{F}_N^{-1}$ 是离散逆傅里叶变换，它们的定义分别为

### 离散傅里叶变换：
对于 $N \in \mathbb{N}$，我们固定一个规则网格 $\{x_j\}_{j \in \mathcal{J}_N}$，其值为

$$
x_j = \frac{2\pi j}{2N + 1}, \tag{B.11}
$$

其中索引 $j \in \mathcal{J}_N$ 属于索引集

$$
\mathcal{J}_N := \{0, \ldots, 2N\}^d. \tag{B.12}
$$

回顾傅里叶波数集合 (2.10)，我们定义离散傅里叶变换 $\mathcal{F}_N: \mathbb{R}^{\mathcal{J}_N} \to \mathbb{C}^{\mathcal{K}_N}$ 为

$$
\widehat v_k :=
\mathcal{F}_N(v)(k) := \frac{1}{(2N + 1)^d} \sum_{j \in \mathcal{J}_N} v_j e^{-2\pi i (j,k)/N}, \tag{B.13}
$$

其逆变换 $\mathcal{F}_N^{-1}: \mathbb{C}^{\mathcal{K}_N} \to \mathbb{R}^{\mathcal{J}_N}$ 为

$$
\mathcal{F}_N^{-1}(\widehat{v})(j) := \sum_{k \in \mathcal{K}_N} \widehat{v}_k e^{2\pi i (j,k)/N}. \tag{B.14}
$$


证明的下一步是利用公式 (2.12) 中投影算子 $\mathcal{G}_N$ 的自然分解，该分解基于离散傅里叶变换 $\mathcal{F}_N \circ P_N$、离散逆傅里叶变换 $\mathcal{F}_N^{-1}$ 以及傅里叶共轭算子 $\hat{\mathcal{G}}_N$，并通过傅里叶神经算子（FNO）来逼近这些算子。

我们首先记

$$
\mathbb{R}^{2\mathcal{K}_N} = (\mathbb{R}^2)^{\mathcal{K}_N} \ (\simeq \mathbb{C}^{\mathcal{K}_N}), \tag{2.13}
$$

为由系数集合 $\{(v_{1,k}, v_{2,k})\}_{k \in \mathcal{K}_N}$ 构成的集合，其中 $v_{\ell,k} \in \mathbb{R}$ 由一个二元组 $(\ell, k)$ 索引，$\ell \in \{1,2\}$，$k \in \mathcal{K}_N$，并将算子 $\mathcal{F}_N \circ P_N$ 视为映射：

$$
\mathcal{F}_N \circ P_N: a \mapsto \{\mathrm{Re}(\hat{a}_k), \mathrm{Im}(\hat{a}_k)\}_{ k \leq N},
$$

输入为 $a \in L^2(\mathbb{T}^d)$，输出 $\{\mathrm{Re}(\hat{a}_k), \mathrm{Im}(\hat{a}_k)\}_{ k \leq N} \in \mathbb{R}^{2\mathcal{K}_N}$ 被视为 $L^2(\mathbb{T}^d; \mathbb{R}^{2\mathcal{K}_N})$ 中的一个常数函数。对该算子的逼近是以下引理的直接推论，该引理在附录 D.2 中已证明：

**引理 2.7** 设 $B > 0$ 且 $N \in \mathbb{N}$ 给定。对于任意 $\varepsilon > 0$，存在一个 FNO $\mathcal{N}: L^2(\mathbb{T}^d) \to L^2(T^d; \mathbb{R}^{2\mathcal{K}_N})$，将 $v \mapsto \{\mathcal{N}(v)_{\ell,k}\}$，其输出函数为常数函数（作为 $x \in \mathbb{T}^d$ 的函数），且满足

$$
\left.
\begin{aligned}
\| \mathrm{Re}(\hat{v}_k) - \mathcal{N}(v)_{1,k}\ \|_{L^\infty} &\leq \varepsilon \\
\| \mathrm{Im}(\hat{v}_k) - \mathcal{N}(v)_{2,k}\ \|_{L^\infty} &\leq \varepsilon
\end{aligned}
\right\} \quad \forall k \in \mathbb{Z}^d,\ k _\infty \leq N,
$$

对所有 $\| v\| _{L^2} \leq B$ 成立，其中 $\hat{v}_k \in \mathbb{C}$ 表示 $v$ 的第 $k$ 个傅里叶系数。

在下一步中，我们使用FNO（傅里叶神经算子）来近似离散逆傅里叶变换 $\mathcal{F}_N^{-1}$。我们回顾一下，FNOs 作用于函数而非常数。因此，为了连接 $\mathcal{F}_N^{-1}$ 和 FNOs，我们将把映射

$$
\mathcal{F}_N^{-1} : [-R, R]^{2\mathcal{K}_N} \subset \mathbb{R}^{2\mathcal{K}_N} \to L^2(\mathbb{T}^d),
$$

解释为一个映射

$$
\mathcal{F}_N^{-1} :
\begin{cases}
L^2(\mathbb{T}^d; [-R, R]^{2\mathcal{K}_N}) \to L^2(\mathbb{T}^d), \\
\{\mathrm{Re}(\widehat{v}_k), \mathrm{Im}(\widehat{v}_k)\}_{|k|\leq N} \mapsto v(x),
\end{cases}
$$

其中输入 $\{\mathrm{Re}(\widehat{v}_k), \mathrm{Im}(\widehat{v}_k)\}_{|k|\leq N} \in [-R, R]^{2\mathcal{K}_N}$ 被识别为 $L^2(\mathbb{T}^d; [-R, R]^{2\mathcal{K}_N})$ 中的一个常值函数。存在一个形如 (2.6) 的 FNO，能够以所需的精度逼近 (D.17)，这是以下引理的直接结果，该引理在附录 D.3 中证明。

**引理 2.8.** 设 $B > 0$ 且 $N \in \mathbb{N}$ 给定。对于所有 $\epsilon > 0$，存在一个 FNO $\mathcal{N}: L^2(\mathbb{T}^d; \mathbb{R}^{2\mathcal{K}_N}) \to L^2(\mathbb{T}^d)$，使得对任意 $v \in L^2_N(\mathbb{T}^d)$ 且 $\|v\|_{L^2} \leq B$，我们有

$$
\|v - \mathcal{N}(w)\|_{L^2} \leq \epsilon,
$$

其中 $w(x) := \{(\mathrm{Re}(\widehat{v}_k), \mathrm{Im}(\widehat{v}_k))\}_{k \in \mathcal{K}_N}$，即 $w \in L^2(\mathbb{T}^d; \mathbb{R}^{2\mathcal{K}_N})$ 是一个收集了 $v$ 的傅里叶系数 $\widehat{v}_k$ 的实部和虚部的常值函数。

最后，通过设定 $\widehat{K} := \mathcal{F}_N(P_N K) \subset \mathbb{C}^{\mathcal{K}_N}$ 作为 $K$ 在连续映射 $\mathcal{F}_N \circ P_N: L^2(\mathbb{T}^d) \to \mathbb{C}^{\mathcal{K}_N}$ 下的（紧致）像，并将 $\mathbb{C}^{\mathcal{K}_N} \simeq \mathbb{R}^{2\mathcal{K}_N}$，其中 $\widehat{v}_{1,k} := \mathrm{Re}(\widehat{v}_k)$ 且 $\widehat{v}_{2,k} := \mathrm{Im}(\widehat{v}_k)$ 对于 $k \in \mathcal{K}_N$，我们可以将 $\widehat{\mathcal{G}}_N$ 视为一个连续映射

$$
\widehat{\mathcal{G}}_N : \widehat{K} \subset \mathbb{R}^{2\mathcal{K}_N} \to \mathbb{R}^{2\mathcal{K}_N},
$$

定义在一个紧致子集上。因此，根据有限维神经网络的通用逼近定理 [1, 13]，可以很容易地证明存在一个仅包含局部权重的 FNO（参见备注 2.3），它能在紧致子集上以所需的精度逼近这个连续映射 $\widehat{G}_N$。

因此，分解 (2.12) 中的每个分量算子都可以通过 FNOs 以所需的精度进行逼近，而通用逼近定理可以通过组合这些 FNOs 并估计所产生的误差得到，详细内容见附录 D.4。