import torch, logging, os
from utils import *

def validation_process_no_real(args, epoch, model, optimizer, valloader_render, previous_best, best_abs_rel):
    model.eval()

    results = {'d1': torch.tensor([0.0]).cuda(args.device), 'd2': torch.tensor([0.0]).cuda(args.device),
               'd3': torch.tensor([0.0]).cuda(args.device),
               'abs_rel': torch.tensor([0.0]).cuda(args.device), 'sq_rel': torch.tensor([0.0]).cuda(args.device),
               'rmse': torch.tensor([0.0]).cuda(args.device),
               'rmse_log': torch.tensor([0.0]).cuda(args.device), 'log10': torch.tensor([0.0]).cuda(args.device),
               'silog': torch.tensor([0.0]).cuda(args.device)}
    nsamples = torch.tensor([0.0]).cuda(args.device)

    for i, sample in enumerate(valloader_render):

        img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample[
            'valid_mask'].cuda(args.device)

        with torch.no_grad():
            pred = model(img).squeeze(1)
            # pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logging.info('==========================================================================================')
    logging.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logging.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
        *tuple([(v / nsamples).item() for v in results.values()])))
    logging.info('==========================================================================================')
    print()

    for k in results.keys():
        if k in ['d1', 'd2', 'd3']:
            previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        else:
            previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'previous_best': previous_best,
    }
    torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

    avg_abs_rel = (results['abs_rel'] / nsamples).item()
    if avg_abs_rel < best_abs_rel:
        best_abs_rel = avg_abs_rel
        best_checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(best_checkpoint, os.path.join(args.save_path, 'best.pth'))
        logging.info(f"New best model saved with RMSE: {best_abs_rel:.3f}")

    return previous_best, best_abs_rel


def validation_process(args, epoch, model, optimizer, valloader_render, valloader_real, previous_best, best_rmse):
    model.eval()

    results = {'d1': torch.tensor([0.0]).cuda(args.device), 'd2': torch.tensor([0.0]).cuda(args.device),
               'd3': torch.tensor([0.0]).cuda(args.device),
               'abs_rel': torch.tensor([0.0]).cuda(args.device), 'sq_rel': torch.tensor([0.0]).cuda(args.device),
               'rmse': torch.tensor([0.0]).cuda(args.device),
               'rmse_log': torch.tensor([0.0]).cuda(args.device), 'log10': torch.tensor([0.0]).cuda(args.device),
               'silog': torch.tensor([0.0]).cuda(args.device)}
    nsamples = torch.tensor([0.0]).cuda(args.device)

    for i, sample in enumerate(valloader_render):

        img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample[
            'valid_mask'].cuda(args.device)

        with torch.no_grad():
            pred = model(img).squeeze(1)
            # pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logging.info('==========================================================================================')
    logging.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logging.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
        *tuple([(v / nsamples).item() for v in results.values()])))
    logging.info('==========================================================================================')
    print()

    for k in results.keys():
        if k in ['d1', 'd2', 'd3']:
            previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        else:
            previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'previous_best': previous_best,
    }
    torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

    avg_rmse = (results['rmse'] / nsamples).item()
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(best_checkpoint, os.path.join(args.save_path, 'best.pth'))
        logging.info(f"New best model saved with RMSE: {best_rmse:.3f}")

    results = {'d1': torch.tensor([0.0]).cuda(args.device), 'd2': torch.tensor([0.0]).cuda(args.device),
               'd3': torch.tensor([0.0]).cuda(args.device),
               'abs_rel': torch.tensor([0.0]).cuda(args.device), 'sq_rel': torch.tensor([0.0]).cuda(args.device),
               'rmse': torch.tensor([0.0]).cuda(args.device),
               'rmse_log': torch.tensor([0.0]).cuda(args.device), 'log10': torch.tensor([0.0]).cuda(args.device),
               'silog': torch.tensor([0.0]).cuda(args.device)}
    nsamples = torch.tensor([0.0]).cuda(args.device)

    for i, sample in enumerate(valloader_real):

        img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample[
            'valid_mask'].cuda(args.device)

        with torch.no_grad():
            pred = model(img).squeeze(1)
            # pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logging.info('------------------------------------------------------------------------------------------')
    logging.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logging.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
        *tuple([(v / nsamples).item() for v in results.values()])))
    logging.info('------------------------------------------------------------------------------------------')
    print()

    return previous_best, best_rmse



def validation_process_DANN_diff(args, epoch, model, model_pretrain, optimizer, valloader_real, valloader_render, previous_best):
    model.eval()
    model_pretrain.eval()

    results = {'d1': torch.tensor([0.0]).cuda(args.device), 'd2': torch.tensor([0.0]).cuda(args.device),
               'd3': torch.tensor([0.0]).cuda(args.device),
               'abs_rel': torch.tensor([0.0]).cuda(args.device), 'sq_rel': torch.tensor([0.0]).cuda(args.device),
               'rmse': torch.tensor([0.0]).cuda(args.device),
               'rmse_log': torch.tensor([0.0]).cuda(args.device), 'log10': torch.tensor([0.0]).cuda(args.device),
               'silog': torch.tensor([0.0]).cuda(args.device)}
    nsamples = torch.tensor([0.0]).cuda(args.device)

    for i, sample in enumerate(valloader_render):

        img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample[
            'valid_mask'].cuda(args.device)

        with torch.no_grad():
            features = model.encoder_forward(img)
            pred = model_pretrain.decoder_forward(img, features).squeeze(1)

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logging.info('==========================================================================================')
    logging.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logging.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
        *tuple([(v / nsamples).item() for v in results.values()])))
    logging.info('==========================================================================================')
    print()

    for k in results.keys():
        if k in ['d1', 'd2', 'd3']:
            previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        else:
            previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'previous_best': previous_best,
    }
    torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

    results = {'d1': torch.tensor([0.0]).cuda(args.device), 'd2': torch.tensor([0.0]).cuda(args.device),
               'd3': torch.tensor([0.0]).cuda(args.device),
               'abs_rel': torch.tensor([0.0]).cuda(args.device), 'sq_rel': torch.tensor([0.0]).cuda(args.device),
               'rmse': torch.tensor([0.0]).cuda(args.device),
               'rmse_log': torch.tensor([0.0]).cuda(args.device), 'log10': torch.tensor([0.0]).cuda(args.device),
               'silog': torch.tensor([0.0]).cuda(args.device)}
    nsamples = torch.tensor([0.0]).cuda(args.device)

    for i, sample in enumerate(valloader_render):

        img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample[
            'valid_mask'].cuda(args.device)

        with torch.no_grad():
            features = model_pretrain.encoder_forward(img)
            pred = model_pretrain.decoder_forward(img, features).squeeze(1)

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logging.info('==========================================================================================')
    logging.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logging.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
        *tuple([(v / nsamples).item() for v in results.values()])))
    logging.info('==========================================================================================')
    print()



    results = {'d1': torch.tensor([0.0]).cuda(args.device), 'd2': torch.tensor([0.0]).cuda(args.device),
               'd3': torch.tensor([0.0]).cuda(args.device),
               'abs_rel': torch.tensor([0.0]).cuda(args.device), 'sq_rel': torch.tensor([0.0]).cuda(args.device),
               'rmse': torch.tensor([0.0]).cuda(args.device),
               'rmse_log': torch.tensor([0.0]).cuda(args.device), 'log10': torch.tensor([0.0]).cuda(args.device),
               'silog': torch.tensor([0.0]).cuda(args.device)}
    nsamples = torch.tensor([0.0]).cuda(args.device)

    for i, sample in enumerate(valloader_real):

        img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample[
            'valid_mask'].cuda(args.device)

        with torch.no_grad():
            features = model.encoder_forward(img)
            pred = model_pretrain.decoder_forward(img, features).squeeze(1)

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logging.info('------------------------------------------------------------------------------------------')
    logging.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logging.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
        *tuple([(v / nsamples).item() for v in results.values()])))
    logging.info('------------------------------------------------------------------------------------------')
    print()
    return previous_best
