import logging, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import rasterio
from rasterio.errors import RasterioIOError
from . import readiness
from .config import settings
from .ml import opensr
from .services import ident, now, save_json, load_json, inspect_raster, make_preview, process, compare_metrics, validate_baseline_output

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger("orbitsrm")

@asynccontextmanager
async def lifespan(_:FastAPI):
    logger.info("%s %s starting: storage=local_filesystem database=not_configured opensr=%s",
                settings.app_name,settings.app_version,"enabled" if settings.opensr_enabled else "disabled")
    yield
    logger.info("%s %s shutting down",settings.app_name,settings.app_version)

app=FastAPI(title=f"{settings.app_name} API",version=settings.app_version,lifespan=lifespan,
            description="OrbitSRM satellite imagery super-resolution and analysis platform. Interpolation is a resampling baseline, not scientifically validated super-resolution.")
app.add_middleware(CORSMiddleware,allow_origins=settings.origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
ALLOWED={".tif",".tiff",".png",".jpg",".jpeg"}

class StartRequest(BaseModel):
    image_id:str
    method:str="baseline"
    scale_factor:int=Field(default=2,ge=2,le=4)
    target_resolution:float|None=Field(default=None,gt=0)
    parameters:dict={}

def run_job(job_id):
    job=load_json("job",job_id)
    try:
        job["status"]="processing"; job["started_at"]=now(); save_json("job",job_id,job)
        result=process(job["image_id"],job)
        job.update(status="completed",completed_at=now(),result=result)
    except Exception as e:
        logging.exception("Processing job %s failed",job_id)
        job.update(status="failed",completed_at=now(),error=str(e))
    save_json("job",job_id,job)

def public_job(job):
    result=job.get("result")
    if result:
        result={k:v for k,v in result.items() if k not in ("output_path","preview_path")}
    return {**job,"result":result} if result is not None else job

@app.get("/api/health")
def health():
    try:
        import rasterio as rio
        rasterio_ok=True; rasterio_version=rio.__version__
    except Exception: rasterio_ok=False; rasterio_version=None
    try: opensr_status=opensr.status(probe_device=False)
    except Exception as e: opensr_status={"available":False,"reason":f"OpenSR status check failed: {e}"}
    return {"status":"ok","application":{"name":settings.app_name,"version":settings.app_version},
            "dependencies":{"rasterio":rasterio_ok,"rasterio_version":rasterio_version},"opensr":opensr_status}

@app.get("/api/ready")
def ready():
    """Readiness probe: 200 only when OrbitSRM can serve its documented endpoints.

    Reports capability labels honestly (local file storage, no database, in-process
    background execution) and never exposes credentials or filesystem paths.
    """
    report=readiness.check(settings,opensr)
    return JSONResponse(status_code=200 if report["status"]=="ready" else 503,content=report)

@app.post("/api/imagery/upload")
async def upload(file:UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED: raise HTTPException(415,"Unsupported file extension; use GeoTIFF, PNG, or JPEG")
    image_id=ident(); path=settings.data_dir/"uploads"/f"{image_id}{suffix}"
    preview=settings.data_dir/"previews"/f"{image_id}.png"
    size=0
    try:
        with path.open("wb") as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>settings.max_upload_mb*1024*1024: raise HTTPException(413,"Upload exceeds configured size limit")
                out.write(chunk)
        if suffix in (".tif",".tiff"):
            meta=inspect_raster(path); kind="geotiff"
        else:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img: meta={"width":img.width,"height":img.height,"bands":len(img.getbands()),"crs":None,"resolution":None,"transform":None,"dtype":"uint8","nodata":None}
            kind="preview_only"
        if meta["width"] > settings.max_raster_width:
            raise HTTPException(413,f"Raster width exceeds configured limit ({settings.max_raster_width} pixels)")
        if meta["height"] > settings.max_raster_height:
            raise HTTPException(413,f"Raster height exceeds configured limit ({settings.max_raster_height} pixels)")
        if meta["width"] * meta["height"] > settings.max_raster_pixels:
            raise HTTPException(413,f"Raster pixel count exceeds configured limit ({settings.max_raster_pixels} pixels)")
        if kind=="geotiff": make_preview(path,preview)
        else:
            from PIL import Image
            Image.open(path).convert("RGB").save(preview)
        item={"id":image_id,"original_filename":Path(file.filename or "image").name,"stored_name":path.name,"size_bytes":size,"kind":kind,"uploaded_at":now(),**meta,"preview_url":f"/api/imagery/{image_id}/preview"}
        save_json("image",image_id,item); return item
    except HTTPException:
        path.unlink(missing_ok=True); preview.unlink(missing_ok=True); raise
    except (RasterioIOError, OSError, ValueError) as e:
        path.unlink(missing_ok=True); preview.unlink(missing_ok=True); raise HTTPException(422,f"Invalid or corrupted image: {e}")

@app.get("/api/imagery/{image_id}")
def get_image(image_id:str):
    x=load_json("image",image_id)
    if not x: raise HTTPException(404,"Image not found")
    return x

@app.get("/api/imagery/{image_id}/preview")
def original_preview(image_id:str):
    x=load_json("image",image_id)
    if not x: raise HTTPException(404,"Image not found")
    p=settings.data_dir/"previews"/f"{image_id}.png"
    return FileResponse(p,media_type="image/png")

def opensr_request_check(req,meta):
    """Cheap configuration-level validation for OpenSR jobs.

    Deliberately avoids importing torch or loading the model in the request path:
    device/CUDA problems are reported by the background job instead.
    """
    report=opensr.status(probe_device=False)
    blocked=not (report["enabled"] and report["checkpoint_exists"] and report["torch_installed"]
                 and report["package_installed"] and report["config_exists"])
    if blocked: raise HTTPException(422,f"OpenSR is not available: {report['reason']}")
    try: window=opensr.configured_window_size(); opensr.configured_sampling_steps()
    except opensr.OpenSRError as exc: raise HTTPException(422,f"OpenSR is not available: {exc}")
    if req.scale_factor!=opensr.SCALE_FACTOR:
        raise HTTPException(422,f"OpenSR supports only a scale factor of {opensr.SCALE_FACTOR}")
    if meta.get("bands")!=opensr.EXPECTED_BANDS:
        raise HTTPException(422,"OpenSR requires a 4-band GeoTIFF ordered "+", ".join(opensr.SENTINEL2_BAND_ORDER))
    if not meta.get("crs"): raise HTTPException(422,"OpenSR requires a georeferenced GeoTIFF; no CRS was found")
    try:
        with rasterio.open(settings.data_dir/"uploads"/meta["stored_name"]) as dataset:
            opensr.validate_band_descriptions(dataset.descriptions)
    except opensr.OpenSRError as exc:
        raise HTTPException(422,str(exc)) from exc
    except (RasterioIOError,OSError) as exc:
        raise HTTPException(422,f"OpenSR source GeoTIFF could not be read: {exc}") from exc
    parameters=req.parameters or {}
    if "window_size" in parameters:
        if parameters["window_size"] is None:
            raise HTTPException(422,"parameters.window_size must be an integer")
        try: window=opensr.configured_window_size(parameters["window_size"])
        except opensr.OpenSRError as exc: raise HTTPException(422,f"OpenSR is not available: {exc}")
    allowed={"sampling_steps":1,"overlap":0,"batch_size":1,"window_size":opensr.MIN_WINDOW}
    unknown=[name for name in parameters if name not in allowed]
    if unknown: raise HTTPException(422,f"Unsupported OpenSR parameters {sorted(unknown)}; allowed parameters are {sorted(allowed)}")
    resource_limits={
        "sampling_steps":(settings.opensr_sampling_steps,settings.opensr_max_sampling_steps),
        "batch_size":(settings.opensr_batch_size,settings.opensr_max_batch_size),
    }
    for name,(default,maximum) in resource_limits.items():
        raw=parameters.get(name,default)
        if isinstance(raw,bool): raise HTTPException(422,f"parameters.{name} must be an integer")
        try: value=int(raw)
        except (TypeError,ValueError): raise HTTPException(422,f"parameters.{name} must be an integer")
        if not isinstance(raw,str) and raw!=value:
            raise HTTPException(422,f"parameters.{name} must be an integer")
        if value<1: raise HTTPException(422,f"parameters.{name} must be >= 1")
        if value>maximum: raise HTTPException(422,f"parameters.{name} must be <= {maximum}")
    if "overlap" in parameters:
        try: overlap=int(parameters["overlap"])
        except (TypeError,ValueError): raise HTTPException(422,"parameters.overlap must be an integer")
        if overlap<0: raise HTTPException(422,"parameters.overlap must be >= 0")
        if overlap>=window:
            raise HTTPException(422,f"parameters.overlap must be smaller than the OpenSR window size ({window})")

@app.post("/api/processing/start",status_code=202)
def start(req:StartRequest,background:BackgroundTasks):
    meta=load_json("image",req.image_id)
    if not meta: raise HTTPException(404,"Image not found")
    if meta["kind"]!="geotiff": raise HTTPException(422,"PNG/JPEG are preview-only; geospatial processing requires GeoTIFF")
    if req.method not in ("baseline","trained_srcnn","opensr_ldsrs2"): raise HTTPException(422,"method must be baseline, trained_srcnn, or opensr_ldsrs2")
    if req.method=="opensr_ldsrs2": opensr_request_check(req,meta)
    if req.method=="baseline":
        try: validate_baseline_output(meta["width"],meta["height"],req.scale_factor)
        except ValueError as exc: raise HTTPException(413,str(exc)) from exc
    if req.target_resolution is not None:
        res=meta.get("resolution")
        if not res or max(res)/req.scale_factor>req.target_resolution:
            raise HTTPException(422,"target_resolution must be achievable with the selected scale factor")
    for p in (settings.data_dir/"metadata").glob("job_*.json"):
        try: existing=__import__("json").loads(p.read_text())
        except Exception: continue
        if (existing.get("status") in ("queued","processing") and existing.get("image_id")==req.image_id
            and existing.get("method")==req.method and existing.get("scale_factor")==req.scale_factor
            and existing.get("target_resolution")==req.target_resolution):
            return public_job(existing)
    jid=ident(); job={"id":jid,"image_id":req.image_id,"method":req.method,"scale_factor":req.scale_factor,"target_resolution":req.target_resolution,"parameters":req.parameters,"status":"queued","created_at":now()}
    save_json("job",jid,job); background.add_task(run_job,jid)
    return job

@app.get("/api/jobs")
def jobs(offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
    items=[]
    for p in (settings.data_dir/"metadata").glob("job_*.json"):
        try: items.append(__import__("json").loads(p.read_text()))
        except Exception: logging.warning("Could not read job metadata %s",p.name)
    items.sort(key=lambda x:x.get("created_at",""),reverse=True)
    return {"items":[public_job(x) for x in items[offset:offset+limit]],"total":len(items),"offset":offset,"limit":limit}

@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    x=load_json("job",job_id)
    if not x: raise HTTPException(404,"Job not found")
    return public_job(x)

@app.get("/api/jobs/{job_id}/result")
def result(job_id:str):
    x=load_json("job",job_id)
    if not x: raise HTTPException(404,"Job not found")
    if x["status"]!="completed": raise HTTPException(409,f"Job is {x['status']}")
    result={k:v for k,v in x["result"].items() if k not in ("output_path","preview_path")}
    return {"job_id":job_id,"image_id":x["image_id"],"method":x["method"],"scale_factor":x["scale_factor"],"result":result,"preview_url":f"/api/jobs/{job_id}/preview","download_url":f"/api/jobs/{job_id}/download"}

@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id:str):
    x=load_json("job",job_id)
    if not x or x.get("status")!="completed": raise HTTPException(404,"Completed job preview not found")
    return FileResponse(x["result"]["preview_path"],media_type="image/png")

@app.get("/api/jobs/{job_id}/download")
def download(job_id:str):
    x=load_json("job",job_id)
    if not x or x.get("status")!="completed": raise HTTPException(404,"Completed job output not found")
    return FileResponse(x["result"]["output_path"],media_type="image/tiff",filename=f"orbitsrm_{job_id}.tif")

@app.delete("/api/jobs/{job_id}",status_code=204)
def delete_job(job_id:str):
    x=load_json("job",job_id)
    if not x: raise HTTPException(404,"Job not found")
    for key in ("output_path","preview_path"):
        if x.get("result",{}).get(key):
            p=Path(x["result"][key]).resolve()
            if p.parent.resolve() in ((settings.data_dir/"outputs").resolve(),(settings.data_dir/"previews").resolve()): p.unlink(missing_ok=True)
    (settings.data_dir/"metadata"/f"job_{job_id}.json").unlink(missing_ok=True)

@app.post("/api/validation/compare")
def validate(output_job_id:str,reference_image_id:str):
    job=load_json("job",output_job_id); ref=load_json("image",reference_image_id)
    if not job or job.get("status")!="completed": raise HTTPException(404,"Completed output job not found")
    if not ref or ref["kind"]!="geotiff": raise HTTPException(404,"GeoTIFF reference image not found")
    try: return {"available":True,"metrics":compare_metrics(job["result"]["output_path"],str(settings.data_dir/"uploads"/ref["stored_name"]))}
    except ValueError as e: raise HTTPException(422,str(e))

MODELS=[{"id":"baseline","name":"Cubic interpolation baseline","architecture":"Rasterio cubic resampling","input_channels":"any supported GeoTIFF","scale_factors":[2,3,4],"trained_weights_available":False,"device_support":["cpu"],"scientific_evaluation_suitable":False,"description":"Pipeline testing baseline. Adds pixels by interpolation; no learned detail."},{"id":"trained_srcnn","name":"Checkpoint SRCNN","architecture":"Residual CNN (SRCNN-style)","input_channels":[1,3],"scale_factors":[2,3,4],"trained_weights_available":bool(settings.model_checkpoint and Path(settings.model_checkpoint).is_file()),"device_support":["cpu","cuda"],"scientific_evaluation_suitable":False,"description":"Requires compatible trained checkpoint; no checkpoint is bundled."}]
MODELS.append(opensr.model_card())
@app.get("/api/models")
def models(): return {"items":MODELS}
@app.get("/api/models/{model_id}")
def model_detail(model_id:str):
    x=next((m for m in MODELS if m["id"]==model_id),None)
    if not x: raise HTTPException(404,"Model not found")
    return x
