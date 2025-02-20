# RingFormer: Rethinking Recurrent Transformer with Adaptive Level Signals

**Jaemu Heo\*, Eldor Fozilov\*, Hyunmin Song, Taehwan Kim**  
IMML Lab, UNIST  
📧 {skek000, eldorfozilov, hyunminsong, taehwankim}@unist.ac.kr  

---

## 📖 Abstract
Transformers have achieved great success in effectively processing sequential data such as text. Their architecture consisting of several attention and feedforward blocks can model relations between elements of a sequence in parallel manner, which makes them very efficient to train and effective in sequence modeling. Even though they have shown strong performance in processing sequential data, the size of their parameters is considerably larger when compared to other architectures such as RNN and CNN based models. Therefore, several approaches have explored parameter sharing and recurrence in Transformer models to address their computational demands. However, such methods struggle to maintain high performance compared to the original transformer model. To address this challenge, we propose our novel approach, RingFormer, which employs one Transformer layer that processes input repeatedly in a circular, ring-like manner, while utilizing low-rank matrices to generate input-dependent level signals. This allows us to reduce the model parameters substantially while maintaining high performance in a variety of tasks such as translation and image classification, as validated in the experiments.

**Contributions:**  
    
    ✅  We enhance a recurrent Transformer architecture to significantly reduce the model's parameter count
        while maintaining high performance.

    ✅  We propose novel input-dependent level signals generated in a parameter-efficient way
        using low-rank matrices to improve the adaptability of a recurrent Transformer model,
        and show that those signals help the model replicate the behavior of the original model.

    ✅  We demonstrate the validity of our approach through careful analysis and ablation studies,
        and show the effectiveness of our model on tasks such as translation and image classification.

For more details, check our paper:  
📄 **[RingFormer: Rethinking Recurrent Transformer with Adaptive Level Signals](https://arxiv.org/abs/2502.13181)**  



