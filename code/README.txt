Run gen_values.py to generate all the data for the final test. And the data will be stored in './data/'

The AMenuNet architecture is in net.py. The AMA mechanism is in auction.py

To reproduce all of our experimental results, run *.sh. The results will be stored in './results/*/'

experiments.py includes train, valid and test parts. But if you have already got the checkpoint and only want to test it, run test.py. 
This may happen when conducting out-of-setting experiments.
One thing to be careful is that to make sure that menu size and $\tau_A$ are the same between the test time and train time.

You can also adjust hyperparameters and different configurations in experiments.py and run it.





