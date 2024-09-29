#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
import numpy as np
from createdataset import create_dataset
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import  create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy
import matplotlib.pyplot as plt
import models


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='image_folder',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='val',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=4,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=True,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')

def validate(args):
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=4,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        name=args.dataset,root=args.data, split=args.split, is_training=False,
        class_map=args.class_map,
        # download=args.dataset_download,
        batch_size=args.batch_size,
        num_classes=args.num_classes)
    # print(len(dataset))


    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        # distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
        # tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    angel_top1 = AverageMeter()
    add_top = AverageMeter()
    add_mae_meter = AverageMeter()
    class_mae_meter = AverageMeter()
    angel_mae_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()
        add_result =[]
        for batch_idx, (input, target, angel_target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
                angel_target = angel_target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output,angel_output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            # loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)
            if batch_idx == 0:
                pre_angel = angel_output
                angel_result = angel_target
                pre_result = output
                target_result = target
            else:
                pre_result = torch.cat((pre_result,output),dim=0)
                target_result = torch.cat((target_result,target),dim=0)
                pre_angel = torch.cat((pre_angel,angel_output),dim=0)
                angel_result = torch.cat((angel_result,angel_target),dim=0)

            output[output>=0.5] = 1
            output[output<0.5] = 0
            predict_class = torch.sum(output,dim=1)[:,0]

            angel_output[angel_output>=0.5] = 1
            angel_output[angel_output<0.5] = 0
            predict_angel = torch.sum(angel_output,dim=1)[:,0]
            
            #add
            predict_angel_ = torch.clone(predict_angel)
            predict_angel_ = predict_angel_.to('cpu')
            predict_angel_ = predict_angel_.numpy()
            predict_angel_ = predict_angel_.astype(np.int32)
         
            predict_class_ = torch.clone(predict_class)
            predict_class_ = predict_class_.to('cpu')
            predict_class_ = predict_class_.numpy()
            predict_class_ = predict_class_.astype(np.int32)
        
            target_c = torch.clone(target)
            target_an = torch.clone(angel_target)
            target_c = torch.sum(target_c,dim=1)[:,0]
            target_an = torch.sum(target_an,dim=1)[:,0]
          
            table = {0:[0,1,2],1:[2,3,4],2:[4,5,6,7,8,9],3:[9]}
            for i in range(len(predict_angel_)):
                if predict_angel_[i]  not in table[predict_class_[i]]:
            
                    if predict_angel_[i] < table[predict_class_[i]][0]:
                        tt = table[predict_class_[i]][0]
                        if tt - predict_angel_[i] <=1:
                            predict_angel_[i] = table[predict_class_[i]][0]
                    else:
                        tt =  table[predict_class_[i]][-1]
                        if abs(predict_angel_[i]-tt)<=1:
                            predict_angel_[i] = table[predict_class_[i]][-1]
                    
     
            for item in predict_angel_:
                add_result.append(item)
            loss = criterion(output, target)+criterion(angel_output,angel_target)
            
            target = torch.sum(target,dim=1)[:,0]
            angel_target = torch.sum(angel_target,dim=1)[:,0]
            
            num_same = (predict_class == target).sum().item()
            absolute_error = abs(predict_class - target)
            absolute_error =  (absolute_error.sum())/target.size(0)
         
            angel_same = torch.tensor((predict_angel == angel_target).sum().item())
            # print('true',angel_same)
            angel_mae = abs(predict_angel-angel_target)
            angel_mae=(angel_mae.sum())/target.size(0)
    
         
            angel_target_ = torch.clone(angel_target)
            angel_target_ = angel_target_.to('cpu')
            angel_target_ = angel_target_.numpy()
            angel_target_ = angel_target_.astype(np.int32)
            same = (predict_angel_ == angel_target_).sum().item()
            add_mae = abs(predict_angel_ - angel_target_)
            add_mae = (add_mae.sum())/target.size(0)
            # print(same)
            
            acc1 = torch.tensor(num_same/len(target)*100)
            angel_acc1 = torch.tensor(angel_same/len(angel_target)*100)
            acc = torch.tensor(same/len(target)*100)
            
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            angel_top1.update(angel_acc1.item(), input.size(0))
            add_top.update(acc.item(),input.size(0))
            add_mae_meter.update(add_mae.item(),input.size(0))
            class_mae_meter.update(absolute_error.item(),input.size(0))
            angel_mae_meter.update(angel_mae.item(),input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {angel_top1.val:>7.3f} ({angel_top1.avg:>7.3f})'
                    'Acc: {add_top.val:>7.3f} ({add_top.avg:>7.3f})'
                    'class_mae: {class_mae_meter.val:>7.3f} ({class_mae_meter.avg:>7.3f})'
                    'angel_mae {angel_mae_meter.val:>7.3f} ({angel_mae_meter.avg:>7.3f})'
                    .format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, angel_top1=angel_top1, add_top=add_top,class_mae_meter = class_mae_meter, angel_mae_meter = angel_mae_meter))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, angel_top1a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, angel_top1a = top1.avg, angel_top1.avg
        add_topa = add_top.avg
        add_maea = add_mae_meter.avg
        class_mae = class_mae_meter.avg
        angel_mae = angel_mae_meter.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        angel_top1=round(angel_top1a, 4), angel_err=round(100 - angel_top1a, 4),
        add_top = round(add_topa,4),add_err=round(100-add_topa,4),
        add_mae = round(add_maea,4),
        class_mae = round(class_mae,4),
        angel_mae = round(angel_mae,4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) angel_Acc@5 {:.3f} ({:.3f}) add_Acc {:.3f} ({:.3f}) add_mae {:.3f} class_mae {:.3f} angel_mae {:.3f}'.format(
       results['top1'], results['top1_err'], results['angel_top1'], results['angel_err'],results['add_top'],results['add_err'],results['add_mae'],results['class_mae'],results['angel_mae']))
 
    return results,pre_result,target_result,pre_angel,angel_result,add_result


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        
        results,pre_result,target_result,pre_angel,angel_result,add_result=validate(args)
        pre_result[pre_result>=0.5] = 1
        pre_result[pre_result<0.5] = 0
        # print(pre_result)
        level = torch.sum(pre_result,dim=1)[:,0]

        target_clone = torch.clone(target_result)
        target_level = torch.sum(target_clone,dim=1)[:,0]
        re,sp,pr,npv,acc,ba,f1,kappa = calculate_metric(target_level,level)
       
        print(f're:{re:.3f} sp:{sp:.3f} pr:{pr:.3f} npv:{npv:.3f} ba:{ba:.3f} f1:{f1:.3f} kappa:{kappa:.3f}')

        
def return_metric(true_labels, predicted_labels, class_label):
    tp = torch.sum((true_labels == class_label) & (predicted_labels == class_label)).item()
    fn = torch.sum((true_labels == class_label) & (predicted_labels != class_label)).item()
    tn = torch.sum((true_labels != class_label) & (predicted_labels != class_label)).item()
    fp = torch.sum((true_labels != class_label) & (predicted_labels == class_label)).item()
    # print(tp+fn+tn+fp)
    total = tp + tn + fp + fn
    Po = (tp + tn) / total
    Pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    kappa = (Po - Pe) / (1 - Pe)

    re = tp/(tp+fn) if (tp+fn)!=0 else 0
    sp = tn/(tn+fp) if (tn+fp)!=0 else 0
    pr = tp/(tp+fp) if (tp+fp)!=0 else 0
    npv = tn/(tn+fn) if (tn+fn)!=0 else 0
    acc = (tp+tn)/(tp+fn+fp+tn) if (tp+fn+fp+tn)!=0 else 0
    ba = (re+sp)/2
    f1 = 2*pr*re/(pr+re) if (pr+re)!=0 else 0
    return [re,sp,pr,npv,acc,ba,f1,kappa]

def ave_list(inlist):
    output = sum(inlist) / len(inlist)
    return output

def calculate_metric(true_labels, predicted_labels):
# 计算每个类别的 Recall
    unique_labels = torch.unique(true_labels)
    re = [return_metric(true_labels, predicted_labels, label)[0] for label in unique_labels]
    sp = [return_metric(true_labels, predicted_labels, label)[1] for label in unique_labels]
    pr = [return_metric(true_labels, predicted_labels, label)[2] for label in unique_labels]
    npv = [return_metric(true_labels, predicted_labels, label)[3] for label in unique_labels]
    acc = [return_metric(true_labels, predicted_labels, label)[4] for label in unique_labels]
    ba = [return_metric(true_labels, predicted_labels, label)[5] for label in unique_labels]
    f1 = [return_metric(true_labels, predicted_labels, label)[6] for label in unique_labels]
    kappa = [return_metric(true_labels, predicted_labels, label)[7] for label in unique_labels]
    # print(recalls)
    re = ave_list(re)
    sp = ave_list(sp)
    pr =  ave_list(pr)
    npv =  ave_list(npv)
    acc =  ave_list(acc)
    ba =  ave_list(ba)
    f1 =  ave_list(f1)
    kappa = ave_list(kappa)
    return re,sp,pr,npv,acc,ba,f1,kappa

def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
