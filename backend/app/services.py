import json, uuid, logging, os
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from PIL import Image
from .config import settings

log = logging.getLogger(__name__)
def now(): return datetime.now(timezone.utc).isoformat()
def ident(): return uuid.uuid4().hex
def save_json(kind, key, obj):
    p = settings.data_dir / "metadata" / f"{kind}_{key}.json"
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return obj
def load_json(kind, key):
    p = settings.data_dir / "metadata" / f"{kind}_{key}.json"
    if not p.exists(): return None
    return json.loads(p.read_text())
def image_path(image_id):
    meta=load_json("image", image_id)
    if not meta: return None
    return settings.data_dir / "uploads" / meta["stored_name"]

def validate_baseline_output(width, height, scale_factor):
    output_width=width*scale_factor
    output_height=height*scale_factor
    output_pixels=output_width*output_height
    if output_pixels>settings.max_baseline_output_pixels:
        raise ValueError(f"Baseline output exceeds configured pixel limit ({settings.max_baseline_output_pixels} pixels)")
    return output_width,output_height

def inspect_raster(path):
    with rasterio.open(path) as ds:
        return {"width":ds.width,"height":ds.height,"bands":ds.count,"crs":str(ds.crs) if ds.crs else None,
          "transform":list(ds.transform)[:6],"resolution":[abs(ds.transform.a),abs(ds.transform.e)],
          "dtype":ds.dtypes[0],"nodata":ds.nodata,"driver":ds.driver}

def make_preview(src, dest, max_size=1400):
    with rasterio.open(src) as ds:
        scale=min(1, max_size/max(ds.width,ds.height))
        w=max(1,int(ds.width*scale)); h=max(1,int(ds.height*scale))
        if ds.count >= 3:
            a=ds.read([1,2,3],out_shape=(3,h,w),resampling=Resampling.nearest).astype("float32")
        else:
            a=ds.read(1,out_shape=(h,w),resampling=Resampling.nearest)[None].astype("float32")
        out=[]
        for band in a:
            good=np.isfinite(band)
            lo,hi=np.percentile(band[good],[2,98]) if good.any() else (0,1)
            out.append(np.clip((band-lo)/max(hi-lo,1e-6)*255,0,255).astype("uint8"))
        rgb=np.stack(out,axis=-1)
        if rgb.shape[-1]==1: rgb=np.repeat(rgb,3,axis=-1)
        Image.fromarray(rgb).save(dest,format="PNG")

def process(image_id, job):
    src=image_path(image_id)
    if not src: raise ValueError("Uploaded image no longer exists")
    method=job["method"]; scale=job["scale_factor"]
    if method=="opensr_ldsrs2":
        # Imported lazily: the module imports this one, and it is only needed for OpenSR jobs.
        from .opensr_pipeline import process as process_opensr
        return process_opensr(src,job)
    with rasterio.open(src) as ds:
        if method=="trained_srcnn":
            try:
                import torch
                from .ml.model import SRCNN
            except ImportError as e:
                raise ValueError("PyTorch is required for trained_srcnn inference; install the optional ML dependencies") from e
            ckpt=settings.model_checkpoint
            if not ckpt or not Path(ckpt).is_file(): raise ValueError("Trained checkpoint is not configured or does not exist")
            if ds.count not in (1,3): raise ValueError("SRCNN checkpoint path supports only 1 or 3 input bands")
            device="cuda" if settings.device=="auto" and torch.cuda.is_available() else ("cpu" if settings.device=="auto" else settings.device)
            model=SRCNN(ds.count).to(device)
            state=torch.load(ckpt,map_location=device,weights_only=True)
            model.load_state_dict(state.get("state_dict",state)); model.eval()
        if method=="baseline":
            out_w,out_h=validate_baseline_output(ds.width,ds.height,scale)
        else:
            out_w,out_h=ds.width*scale,ds.height*scale
        profile=ds.profile.copy()
        # Input block sizes describe the source and may be illegal for the scaled output.
        profile.update(width=out_w,height=out_h,transform=ds.transform*Affine.scale(1/scale),
                       compress="deflate",tiled=True,blockxsize=256,blockysize=256)
        out_path=settings.data_dir/"outputs"/f"{job['id']}.tif"
        # Rasterio performs bounded-memory window reads; upscale each source window into output windows.
        tile=512
        with rasterio.open(out_path,"w",**profile) as dst:
            for y in range(0,ds.height,tile):
                for x in range(0,ds.width,tile):
                    hh=min(tile,ds.height-y); ww=min(tile,ds.width-x)
                    win=rasterio.windows.Window(x,y,ww,hh)
                    block=ds.read(window=win,out_shape=(ds.count,hh*scale,ww*scale),resampling=Resampling.cubic).astype("float32")
                    if method=="trained_srcnn":
                        # Model operates on normalized values, preserving approximate source radiometry.
                        lo=float(np.nanmin(block)); hi=float(np.nanmax(block)); span=max(hi-lo,1e-6)
                        inp=torch.from_numpy(np.nan_to_num((block-lo)/span)[None]).to(device)
                        with torch.no_grad(): block=(model(inp).cpu().numpy()[0]*span+lo)
                    dst.write(block.astype(ds.dtypes[0]),window=rasterio.windows.Window(x*scale,y*scale,ww*scale,hh*scale))
        preview=settings.data_dir/"previews"/f"{job['id']}.png"; make_preview(out_path,preview)
    return {"output_path":str(out_path),"preview_path":str(preview),"output":inspect_raster(out_path),"created_at":now(),
      "method":method,"trained_model_used":method=="trained_srcnn","inferred_details":True,
      "quantitative_uncertainty":"unavailable","validation":"not performed; no reference image supplied",
      "limitations":["Output pixels are resampled/inferred and do not represent newly observed ground detail.","No calibrated uncertainty estimate is available."]}

def compare_metrics(pred_path, ref_path):
    from skimage.metrics import structural_similarity
    with rasterio.open(pred_path) as a, rasterio.open(ref_path) as b:
        if a.crs!=b.crs or a.width!=b.width or a.height!=b.height or a.count!=b.count or a.transform!=b.transform:
            raise ValueError("Rasters must have matching CRS, transform, dimensions, and band count")
        width,height=a.width,a.height
        chunk_size=settings.metric_chunk_size
        squared_error=0.0
        valid_count=0
        valid_ref_min=None
        valid_ref_max=None
        ref_non_nan_min=None
        ref_non_nan_max=None
        for y0 in range(0,height,chunk_size):
            chunk_height=min(chunk_size,height-y0)
            for x0 in range(0,width,chunk_size):
                chunk_width=min(chunk_size,width-x0)
                window=rasterio.windows.Window(x0,y0,chunk_width,chunk_height)
                pred_chunk=a.read(window=window).astype(np.float64)
                ref_chunk=b.read(window=window).astype(np.float64)
                valid=np.isfinite(pred_chunk)&np.isfinite(ref_chunk)
                differences=None
                valid_values=None
                non_nan_ref=None
                if valid.any():
                    differences=pred_chunk[valid]-ref_chunk[valid]
                    squared_error+=float(np.sum(differences*differences,dtype=np.float64))
                    valid_count+=int(np.count_nonzero(valid))
                    valid_values=ref_chunk[valid]
                    local_min=float(np.min(valid_values))
                    local_max=float(np.max(valid_values))
                    valid_ref_min=local_min if valid_ref_min is None else min(valid_ref_min,local_min)
                    valid_ref_max=local_max if valid_ref_max is None else max(valid_ref_max,local_max)
                non_nan_ref=ref_chunk[~np.isnan(ref_chunk)]
                if non_nan_ref.size:
                    local_min=float(np.min(non_nan_ref))
                    local_max=float(np.max(non_nan_ref))
                    ref_non_nan_min=local_min if ref_non_nan_min is None else min(ref_non_nan_min,local_min)
                    ref_non_nan_max=local_max if ref_non_nan_max is None else max(ref_non_nan_max,local_max)
                del pred_chunk,ref_chunk,valid,differences,valid_values,non_nan_ref
        if valid_count==0: raise ValueError("No valid overlapping pixels")
        if valid_ref_min is None or valid_ref_max is None:
            raise ValueError("No valid overlapping pixels")
        rmse=float(np.sqrt(squared_error/valid_count))
        peak=float(valid_ref_max-valid_ref_min)
        psnr=float("inf") if rmse==0 else float(20*np.log10(max(peak,1e-12)/rmse))

        win=min(7,min(height,width))
        if win%2==0: win-=1
        if win<3: raise ValueError("Images are too small for reliable SSIM (minimum 3x3)")

        if ref_non_nan_min is None or ref_non_nan_max is None:
            ref_range=float("nan")
        else:
            ref_range=float(ref_non_nan_max-ref_non_nan_min)
        data_range=max(float(ref_range),1e-12)
        pad=(win-1)//2
        ssim_sum=0.0
        ssim_count=0
        for core_y in range(pad,height-pad,chunk_size):
            core_height=min(chunk_size,height-pad-core_y)
            read_y=core_y-pad
            read_height=core_height+2*pad
            for core_x in range(pad,width-pad,chunk_size):
                core_width=min(chunk_size,width-pad-core_x)
                read_x=core_x-pad
                read_width=core_width+2*pad
                window=rasterio.windows.Window(read_x,read_y,read_width,read_height)
                pred_chunk=np.moveaxis(a.read(window=window).astype(np.float64),0,-1)
                ref_chunk=np.moveaxis(b.read(window=window).astype(np.float64),0,-1)
                _,ssim_map=cast(tuple[float,np.ndarray],structural_similarity(
                    pred_chunk,ref_chunk,channel_axis=-1,data_range=data_range,win_size=win,full=True))
                core_ssim=ssim_map[pad:pad+core_height,pad:pad+core_width,:]
                ssim_sum+=float(np.sum(core_ssim,dtype=np.float64))
                ssim_count+=core_ssim.size
                del pred_chunk,ref_chunk,ssim_map,core_ssim
        ssim=float(ssim_sum/ssim_count)
    return {"rmse":rmse,"psnr":psnr,"ssim":ssim}
