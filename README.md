先写成中文   
NonLlama: 不是Llama但是有点像的小项目   

## Write at the beginning


这不是一个完全从0开始(from scratch)的项目，相同类型的项目已经有很多，同时完成这个项目的过程中也从很多大佬的项目中学习(抄)了很多，本项目仅是拾人牙慧而已   
如果想学习和尝试从零实现，以下的项目更加推荐  
- [GitHub - karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.](https://github.com/karpathy/nanoGPT/) 👈 强烈推荐
- [GitHub - DLLXW/baby-llama2-chinese: 用于从头预训练+SFT一个小参数量的中文LLaMa2的仓库；24G单卡即可运行得到一个具备简单中文问答能力的chat-llama2.](https://github.com/DLLXW/baby-llama2-chinese) 
- [GitHub - naklecha/llama3-from-scratch: llama3 implementation one matrix multiplication at a time](https://github.com/naklecha/llama3-from-scratch)

## Introduction

本项目原目标是：从代码层面理解大语言模型的架构和训练   
以下是可能需要知道的关于项目的概要  
- 项目主要基于karpathy大神的 nanoGPT 实现，参考和借鉴了许多其它源码和项目
- 项目实现了一个简化版的LLM Pre-trainning 的过程，粗糙简单地实现了 从数据选择，数据处理，模型架构，多卡训练的过程
	- 项目使用的数据集为 wikipedia-en [wikimedia/wikipedia · Datasets at Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia)
	- 简单地使用了MinHash对数据进行去重
	- 模型架构为类Llama架构，简单实现了 RMSNorm，RoPE和GQA
	- 重新实现了RandomSampler 和 DistributedRandomSampler 以避免大数据集进行shuffle时产生的OOM
	- 使用DDP进行多卡训练

## Additional

- 附带学习nanoGPT时记录的笔记
- 进行中的 [一份从llama3.1开始的梳理](https://emisaber.github.io/White_Box/Notes/%E4%B8%80%E4%BB%BD%E4%BB%8Ellama3.1%E5%BC%80%E5%A7%8B%E7%9A%84%E6%A2%B3%E7%90%86)  

## TODO

- [ ] 数据处理(过滤，去重，数据配比)   
- [ ] 更多的实验来验证代码
- [ ] 更简单的分布式训练方法
- [ ] 后训练


