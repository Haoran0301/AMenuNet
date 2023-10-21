# AMenuNet
This is the code for the paper "A Scalable Neural Network for DSIC Affine Maximizer" in NeurIPS 2023.

```
@article{duan2023scalable,
  title={A Scalable Neural Network for DSIC Affine Maximizer Auction Design},
  author={Duan, Zhijian and Sun, Haoran and Chen, Yurong and Deng, Xiaotie},
  journal={arXiv preprint arXiv:2305.12162},
  year={2023}
}
```
### Generate test data (optional)
Run gen_values.py to generate all the data for the final test. And then you can find the data in './data/'

### Architecture and Mechanism
The architecture of AMenuNet is in net.py. The AMA mechanism is in auction.py

### Training
To reproduce all of our experimental results, run x.sh. The results will be stored in './results/x/'
For example, 

experiments.py includes training, validdation and testing. But if you have already got the checkpoint and only want to test it, run test.py. This may happens when conducting out-of-setting experiments.
One thing to care is that to make sure that menu size and $\tau_A$ are the same between the testing time and training time.

You can also adjust hyperparameters and different configurations in experiments.py and run it.
