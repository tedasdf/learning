- introducing diffusion model trained from scratch unde the pre-training and SFT 



The core of LLM is to capture the true but unkwon lanugae distributoni by optmizing a model dsitribution through maximum likelihood estimation : 


$$
\max_\theta \mathbb{E}_{x \sim p_\text{data}(x)} \big[\log p_\theta(x)\big] \iff \min_\theta \text{KL}\big(p_\text{data}(x) \, \| \, p_\theta(x)\big)
$$

and the paradigm of AM is 'next token prediction : 

$$
p_\theta(x) = p_\theta(x_1) \prod_{i=2}^{L} p_\theta(x_i \mid x_1, \dots, x_{i-1})
\quad \text{(Autoregressive formulation)}
$$


A fundamental question has to be asked : 

**IS the autoregressive paradigm the only path to achieving the core capabilities of LLMs such as scalability , in-context learnign and instruction-following ??**



scalaribilitiy is a consequenc of the interplay between transformers , model size , data sze and fisher consistency incude by the genrative principles . 


The core of LLaDA is a mask predictor, a parametric model pθ(·|xt) that takes xt as input and
predicts all masked tokens (denoted as M) simultaneously. It is trained using a cross-entropy loss
computed only on the masked tokens :

$$
\mathcal{L}(\theta) \coloneqq 
- \mathbb{E}_{t, x_0, x_t} \Bigg[
\frac{1}{t} \sum_{i=1}^{L} 
\mathbf{1}[x_i^t = M] \, 
\log p_\theta(x_i^0 \mid x_t)
\Bigg]
$$
