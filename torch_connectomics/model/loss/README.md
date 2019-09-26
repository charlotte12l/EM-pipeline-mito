# Loss

Implementation of the following loss:

### Multi-resolution Output Loss
- AuxMSELoss: get the WeightedMSELoss of the seg_out + WeightedMSELoss of the aux_out
- AuxBCELoss: get WeightedBCELoss of the seg_out + WeightedBCELoss of the aux_out

### Online Hard Negative Mining Loss
MSE:
- OhemMSE: mean-squared error for OHEM. Returns a matrix rather than a number. Will be used in OhemMSELoss
- OhemWeightedMSE: Weighted mean-squared error for OHEM. Returns a matrix rather than a number. Will be used in OhemMSELoss2
- OhemMSELoss: Online hard negative mining, select the top n weighted MSE Loss
- OhemMSELoss2: Online hard negative mining, select the top n MSE Loss
- AuxOhemMSELoss: Weighted mse on auxiliary output + online hard negative mining on final output
- AuxOhemMSELoss2: Weighted mse on final output + online hard negative mining on auxiliary output

BCE:
- OhemWeightedBCE: Weighted binary cross-entropy for OhemBCELoss. Returns a matrix rather than a number. Will be used in OhemBCELoss
- OhemBCELoss: Online hard negative mining, use threshold and minkeep to select the top losses
- AuxOhemBCELoss: Weighted bce on Auxiliary Output + online hard negative mining on final output

### Discriminative Loss
- DiscriminativeLoss: Discriminative Loss implementation of the [paper](https://arxiv.org/pdf/1708.02551.pdf). Ignore the input where there are only one class( or they can't cluster)
