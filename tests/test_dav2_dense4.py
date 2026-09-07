import copy
import math

import numpy as np
import pytest
import torch

from bim_priorda3.data.dense4 import augment_dense4
from bim_priorda3.models.dav2_dense4 import build_dense4_condition, dense_silog_loss


def batch():
    return {"base_depth":torch.full((2,1,4,4),2.), "bim_depth":torch.full((2,1,4,4),6.), "bim_valid":torch.ones(2,1,4,4)}


def test_condition_preserves_metric_and_global_q_information():
    b=batch()
    c=build_dense4_condition(b)
    assert c.shape==(2,4,4,4)
    torch.testing.assert_close(c[:,0],torch.full_like(c[:,0],math.log(2)))
    torch.testing.assert_close(c[:,1],torch.full_like(c[:,1],math.log(6)))
    torch.testing.assert_close(c[:,2],torch.full_like(c[:,2],math.log(3)))
    b["base_depth"]=b["base_depth"]*math.exp(.2)
    perturbed=build_dense4_condition(b)
    torch.testing.assert_close(perturbed[:,0],c[:,0]+.2)
    torch.testing.assert_close(perturbed[:,2],c[:,2]-.2)
    torch.testing.assert_close(perturbed[:,1],c[:,1])
    torch.testing.assert_close(perturbed[:,3],c[:,3])


def test_missing_bim_keeps_da3_and_handles_invalid_logs():
    b=batch()
    b["bim_valid"].zero_()
    b["bim_depth"].fill_(float("nan"))
    c=build_dense4_condition(b)
    assert torch.isfinite(c).all()
    assert torch.count_nonzero(c[:,1:])==0
    torch.testing.assert_close(c[:,0],b["base_depth"][:,0].log())


def test_no_disagreement_clip_and_gradients():
    b=batch()
    b["base_depth"].requires_grad_()
    b["bim_depth"]*=100
    c=build_dense4_condition(b)
    assert c[:,2].min()>1.5
    c[:,0].sum().backward()
    assert b["base_depth"].grad.abs().sum()>0
    b["base_depth"]=torch.zeros_like(b["base_depth"])
    with pytest.raises(ValueError):
        build_dense4_condition(b)


def test_silog_official_sample_variance_and_scale_sensitivity():
    gt=torch.tensor([1.,2.,3.,4.]).view(1,1,2,2)
    pred=torch.tensor([1.1,1.9,3.4,3.8]).view_as(gt).requires_grad_()
    mask=torch.ones_like(gt)
    result=dense_silog_loss(pred,gt,mask)
    g=torch.log(pred+1e-7)-torch.log(gt+1e-7)
    expected=10*torch.sqrt(torch.var(g)+.15*torch.mean(g)**2)
    torch.testing.assert_close(result["total"],expected)
    result["total"].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
    scaled=dense_silog_loss(2*gt,gt,mask)["total"]
    assert scaled.item()==pytest.approx(10*math.sqrt(.15)*math.log(2),rel=1e-5)


def test_silog_does_not_discard_bad_predictions_or_gt_above20():
    gt=torch.tensor([1.,30.]).view(1,1,1,2)
    mask=torch.ones_like(gt)
    assert dense_silog_loss(torch.ones_like(gt),gt,mask)["total"]>10
    with pytest.raises(FloatingPointError):
        dense_silog_loss(torch.full_like(gt,float("nan")),gt,mask)


def aug_config(**overrides):
    return {"color_jitter":0.,"bim_dropout_probability":0.,"bim_dropout_fraction":.12,
            "bim_full_dropout_probability":0.,"horizontal_flip_probability":0.,**overrides}


def arrays():
    a={k:np.ones((1,4,4),np.float32) for k in ("base_depth","bim_depth","bim_valid","gt_depth","gt_valid")}
    a["rgb"]=np.full((3,4,4),.25,np.float32)
    return a


def test_shuffle_then_dropout_does_not_restore_bim_or_change_gt():
    a=arrays()
    before=copy.deepcopy(a)
    donor=(np.full((1,4,4),5.,np.float32),np.ones((1,4,4),np.float32))
    output,flags=augment_dense4(a,aug_config(bim_full_dropout_probability=1.),donor=donor)
    assert flags["bim_shuffled"] and flags["bim_full_dropped"]
    assert np.count_nonzero(output["bim_depth"])==np.count_nonzero(output["bim_valid"])==0
    for key in ("base_depth","gt_depth","gt_valid","rgb"):
        np.testing.assert_array_equal(output[key],before[key])
    assert np.all(donor[0]==5.)


def test_shuffle_replaces_both_depth_and_mask():
    donor=(np.full((1,4,4),5.,np.float32),np.ones((1,4,4),np.float32))
    donor[0][...,0,:]=0
    donor[1][...,0,:]=0
    output,_=augment_dense4(arrays(),aug_config(),donor=donor)
    np.testing.assert_array_equal(output["bim_depth"],donor[0])
    np.testing.assert_array_equal(output["bim_valid"],donor[1])


def test_horizontal_flip_applies_to_all_modalities():
    a=arrays()
    for k in a:
        a[k][...,0]=0
    before=copy.deepcopy(a)
    result,flags=augment_dense4(a,aug_config(horizontal_flip_probability=1.))
    assert flags["flipped"]
    for k in a:
        np.testing.assert_array_equal(result[k],before[k][...,::-1])
