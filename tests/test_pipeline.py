"""CPU checks for TSFD loading, loss, optimization and resume behavior."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torchio
from torch import nn
from data.dataset import TSFDDataset, preprocess_slice
from training.engine import create_optimizer, load_checkpoint, run_epoch, save_checkpoint
from training.loss import TSFDLoss
from training.metrics import compute_map_metrics


class PipelineTests(unittest.TestCase):
    def test_brainweb_axes_and_padding(self):
        volume=torch.arange(256.).reshape(1,8,8,4)
        target=torch.arange(64.).reshape(1,8,8,1)
        inputs,maps,mask=preprocess_slice(volume,target,target*2,'brainweb',2)
        expected=volume[0,:,:,:2].permute(2,0,1)
        expected=(expected-expected.min())/(expected.max()-expected.min()+1e-8)
        self.assertTrue(torch.equal(inputs[:2],expected))
        self.assertEqual(tuple(inputs.shape),(400,8,8))
        self.assertEqual(torch.count_nonzero(inputs[2:]).item(),0)
        self.assertTrue(torch.equal(maps[0],maps[1]))
        self.assertTrue(torch.all(mask==1))

    def test_delics_scaling_and_mask(self):
        volume=torch.ones(4,1,8,8)*2
        t1=torch.ones(1,1,8,8)*2500; t2=torch.ones_like(t1)*6000
        mask=torch.ones_like(t1); mask[:,:,0]=0
        inputs,maps,roi=preprocess_slice(volume,t1,t2,'delics',2,mask)
        self.assertTrue(torch.allclose(inputs[:2],torch.ones_like(inputs[:2])))
        self.assertTrue(torch.all(maps[0]==.5) and torch.all(maps[1]==1))
        self.assertEqual(torch.count_nonzero(inputs[2:]).item(),0)
        self.assertTrue(torch.equal(roi,mask[0]))

    def test_real_nifti_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            torchio.ScalarImage(tensor=torch.arange(256.).reshape(1,8,8,4)).save(root/'input.nii.gz')
            for name in ['t1','t2']:
                torchio.ScalarImage(tensor=torch.arange(64.).reshape(1,8,8,1)).save(root/(name+'.nii.gz'))
            with (root/'slices.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['split','input_path','t1_path','t2_path','slice_index'])
                writer.writerow(['train','input.nii.gz','t1.nii.gz','t2.nii.gz',7])
                writer.writerow(['test','input.nii.gz','t1.nii.gz','t2.nii.gz',8])
            dataset=TSFDDataset(root/'slices.csv','invivo','train',2)
            self.assertEqual(len(dataset),1)
            self.assertEqual(dataset[0]['slice_index'],7)
            self.assertEqual(tuple(dataset[0]['inputs'].shape),(400,8,8))
            with self.assertRaises(ValueError):TSFDDataset(root/'slices.csv','invivo','valid')

    def test_loss_gradient_and_mask_denominator(self):
        prediction=torch.ones(1,2,2,2,requires_grad=True); target=torch.zeros_like(prediction)
        mask=torch.tensor([[[[1.,0.],[0.,0.]]]])
        criterion=TSFDLoss()
        self.assertEqual(criterion(prediction,target).item(),1.)
        loss=criterion(prediction,target,mask)
        self.assertEqual(loss.item(),.25)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad,mask.expand_as(prediction)/8))
        with self.assertRaises(ValueError):criterion(prediction,target[:,:1])

    def test_delics_nifti_channel_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            images={'input':torch.ones(4,1,8,8)*2, 't1':torch.ones(1,1,8,8)*2500,
                    't2':torch.ones(1,1,8,8)*2000, 'mask':torch.ones(1,1,8,8)}
            for name,tensor in images.items():torchio.ScalarImage(tensor=tensor).save(root/(name+'.nii.gz'))
            with (root/'slices.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['split','input_path','t1_path','t2_path','mask_path','slice_index'])
                writer.writerow(['test','input.nii.gz','t1.nii.gz','t2.nii.gz','mask.nii.gz',1])
            sample=TSFDDataset(root/'slices.csv','delics','test',2)[0]
            self.assertEqual(tuple(sample['inputs'].shape),(400,8,8))
            self.assertTrue(torch.all(sample['targets']==.5))

    def test_training_and_resume(self):
        model=nn.Conv2d(400,2,1)
        nn.init.zeros_(model.weight); nn.init.zeros_(model.bias)
        inputs=torch.zeros(1,400,8,8); inputs[:,0]=1
        batch={'inputs':inputs,'targets':torch.ones(1,2,8,8)*.5,'mask':torch.ones(1,1,8,8)}
        device=torch.device('cpu'); optimizer,scheduler=create_optimizer(model,.05,epochs=5)
        before=run_epoch(model,[batch],device)
        run_epoch(model,[batch],device,optimizer)
        after=run_epoch(model,[batch],device)
        self.assertLess(after,before)
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt'
            save_checkpoint(path,model,optimizer,scheduler,1,{'dataset':'brainweb'},best_loss=after)
            clone=nn.Conv2d(400,2,1); opt2,sched2=create_optimizer(clone,epochs=5)
            info=load_checkpoint(path,clone,device,opt2,sched2)
            self.assertEqual(info['epoch'],1)
            self.assertTrue(torch.equal(clone(inputs),model(inputs)))
            self.assertEqual(opt2.param_groups[0]['lr'],optimizer.param_groups[0]['lr'])
            self.assertEqual(sched2.state_dict(),scheduler.state_dict())
            run_epoch(model,[batch],device,optimizer); run_epoch(clone,[batch],device,opt2)
            self.assertTrue(torch.equal(model(inputs),clone(inputs)))
            raw=Path(directory)/'weights.pt'
            torch.save({'module.'+k:v for k,v in model.state_dict().items()},raw)
            load_checkpoint(raw,clone,device)
            self.assertTrue(torch.equal(model(inputs),clone(inputs)))

    def test_masked_metrics_units(self):
        reference=np.ones((8,8)); prediction=reference*.5; mask=np.ones((8,8)); mask[:,4:]=0
        first=compute_map_metrics(prediction,reference,mask,5000.)
        second=compute_map_metrics(prediction,reference,mask,4000.)
        self.assertAlmostEqual(first['nmse'],.25)
        self.assertEqual(first['mae'],2500.);self.assertEqual(second['mae'],2000.)
        self.assertTrue(0<first['ssim']<1)
        with self.assertRaises(ValueError):compute_map_metrics(prediction,reference,np.zeros_like(mask))

    def test_bad_frames_and_constant_maps(self):
        volume=torch.ones(1,8,8,2);target=torch.ones(1,8,8,1)
        with self.assertRaises(ValueError):preprocess_slice(volume,target,target,'brainweb',4)
        inputs,maps,_=preprocess_slice(volume,target,target,'brainweb',2)
        self.assertTrue(torch.isfinite(inputs).all() and torch.isfinite(maps).all())
        self.assertEqual(torch.count_nonzero(maps).item(),0)


if __name__=='__main__':
    unittest.main()
