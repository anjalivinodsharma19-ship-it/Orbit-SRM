import io
import numpy as np
import pytest
import rasterio
import shutil
from typing import cast
from pydantic import ValidationError
from rasterio.transform import from_origin
from fastapi.testclient import TestClient
from fastapi import HTTPException
from app import main as main_module
from app.config import Settings, settings
from app.main import app

client=TestClient(app)

@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path,monkeypatch):
    data_dir=tmp_path/"data"
    for folder in ("uploads","outputs","previews","metadata","temporary"):
        (data_dir/folder).mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(settings,"data_dir",data_dir)

def raster_bytes(width=8,height=6):
    mem=io.BytesIO()
    with rasterio.io.MemoryFile() as mf:
        with mf.open(driver="GTiff",width=width,height=height,count=3,dtype="uint16",crs="EPSG:32643",transform=from_origin(500000,2000000,10,10)) as ds: ds.write(np.ones((3,height,width),dtype="uint16")*100)
        return mf.read()

def upload_raster(width=8,height=6):
    return client.post('/api/imagery/upload',files={'file':('tiny.tif',raster_bytes(width,height),'image/tiff')})

def write_metric_raster(path,data,crs="EPSG:32643",transform=None,nodata=None):
    if transform is None:
        transform=from_origin(500000,2000000,10,10)
    with rasterio.open(path,"w",driver="GTiff",width=data.shape[2],height=data.shape[1],
                       count=data.shape[0],dtype=data.dtype,crs=crs,transform=transform,
                       nodata=nodata) as dataset:
        dataset.write(data)
    return path

def full_array_metric_reference(pred_path,ref_path):
    from skimage.metrics import structural_similarity

    with rasterio.open(pred_path) as pred,rasterio.open(ref_path) as ref:
        x=pred.read().astype(np.float64)
        y=ref.read().astype(np.float64)
    valid=np.isfinite(x)&np.isfinite(y)
    if not valid.any():
        raise ValueError("No valid overlapping pixels")
    differences=x[valid]-y[valid]
    rmse=float(np.sqrt(np.mean(differences*differences)))
    peak=float(np.max(y[valid])-np.min(y[valid]))
    psnr=float("inf") if rmse==0 else float(20*np.log10(max(peak,1e-12)/rmse))
    xx=np.moveaxis(x,0,-1)
    yy=np.moveaxis(y,0,-1)
    data_range=max(float(np.nanmax(yy)-np.nanmin(yy)),1e-12)
    win=min(7,min(xx.shape[0],xx.shape[1]))
    if win%2==0:
        win-=1
    if win<3:
        raise ValueError("Images are too small for reliable SSIM (minimum 3x3)")
    ssim=float(cast(float,structural_similarity(xx,yy,channel_axis=-1,data_range=data_range,win_size=win)))
    return {"rmse":rmse,"psnr":psnr,"ssim":ssim}

def assert_metrics_match(actual,expected):
    assert set(actual)=={"rmse","psnr","ssim"}
    for metric in actual:
        np.testing.assert_allclose(actual[metric],expected[metric],rtol=1e-12,atol=1e-12,equal_nan=True)

def test_health_and_models():
    assert client.get('/api/health').status_code==200
    assert client.get('/api/models').json()['items'][0]['id']=='baseline'

def test_upload_and_baseline_job():
    up=client.post('/api/imagery/upload',files={'file':('tiny.tif',raster_bytes(),'image/tiff')})
    assert up.status_code==200,up.text
    image=up.json(); assert image['crs']=='EPSG:32643' and image['bands']==3
    assert client.get('/api/imagery/'+image['id']).status_code==200
    response=client.post('/api/processing/start',json={'image_id':image['id'],'method':'baseline','scale_factor':2})
    assert response.status_code==202,response.text
    job=client.get('/api/jobs/'+response.json()['id']).json()
    assert job['status']=='completed',job
    downloaded=client.get('/api/jobs/'+job['id']+'/download')
    assert downloaded.status_code==200
    with rasterio.MemoryFile(downloaded.content) as mem, mem.open() as ds:
        assert (ds.width,ds.height)==(16,12)
        assert ds.transform.a==5 and ds.crs.to_string()=='EPSG:32643'
    disposition=downloaded.headers['content-disposition']
    assert disposition==f'attachment; filename="orbitsrm_{job["id"]}.tif"',disposition

def test_reject_extension_and_preview_processing():
    bad=client.post('/api/imagery/upload',files={'file':('x.exe',b'x','application/octet-stream')})
    assert bad.status_code==415
    img=client.post('/api/imagery/upload',files={'file':('x.png',b'not an image','image/png')})
    assert img.status_code==422

@pytest.mark.parametrize(("limit_name","limit_value","width","height","detail"),[
    ("max_raster_width",7,8,6,"width"),
    ("max_raster_height",5,8,6,"height"),
    ("max_raster_pixels",47,8,6,"pixel count"),
])
def test_upload_rejects_raster_over_configured_limits(limit_name,limit_value,width,height,detail,monkeypatch):
    monkeypatch.setattr(settings,limit_name,limit_value)
    response=upload_raster(width,height)
    assert response.status_code==413
    assert detail in response.json()["detail"]
    assert list((settings.data_dir/"uploads").iterdir())==[]
    assert list((settings.data_dir/"previews").iterdir())==[]

def test_upload_accepts_raster_exactly_at_configured_limits(monkeypatch):
    monkeypatch.setattr(settings,"max_raster_width",8)
    monkeypatch.setattr(settings,"max_raster_height",6)
    monkeypatch.setattr(settings,"max_raster_pixels",48)
    response=upload_raster()
    assert response.status_code==200,response.text
    assert response.json()["width"]==8 and response.json()["height"]==6

def test_baseline_request_below_output_limit_succeeds(monkeypatch):
    monkeypatch.setattr(settings,"max_baseline_output_pixels",17)
    image=upload_raster(2,2).json()
    response=client.post('/api/processing/start',json={"image_id":image["id"],"method":"baseline","scale_factor":2})
    assert response.status_code==202,response.text
    assert client.get(f'/api/jobs/{response.json()["id"]}').json()["status"]=="completed"

@pytest.mark.parametrize("scale_factor",[2,3,4])
def test_baseline_request_exactly_at_output_limit_accepts_scale_factors(scale_factor,monkeypatch):
    output_pixels=(2*scale_factor)*(2*scale_factor)
    monkeypatch.setattr(settings,"max_baseline_output_pixels",output_pixels)
    image=upload_raster(2,2).json()
    response=client.post('/api/processing/start',json={"image_id":image["id"],"method":"baseline","scale_factor":scale_factor})
    assert response.status_code==202,response.text
    job=client.get(f'/api/jobs/{response.json()["id"]}').json()
    assert job["status"]=="completed"
    with rasterio.open(settings.data_dir/"outputs"/f"{job['id']}.tif") as output:
        assert output.width*output.height==output_pixels

def test_baseline_request_one_pixel_over_limit_is_rejected_before_scheduling(monkeypatch):
    monkeypatch.setattr(settings,"max_baseline_output_pixels",15)
    image=upload_raster(2,2).json()
    response=client.post('/api/processing/start',json={"image_id":image["id"],"method":"baseline","scale_factor":2})
    assert response.status_code==413
    assert "pixel limit" in response.json()["detail"]
    assert list((settings.data_dir/"metadata").glob("job_*.json"))==[]
    assert list((settings.data_dir/"outputs").iterdir())==[]

def test_baseline_output_limit_preserves_missing_and_invalid_input_responses():
    missing=client.post('/api/processing/start',json={"image_id":"missing","method":"baseline","scale_factor":2})
    invalid=client.post('/api/processing/start',json={"image_id":"missing","method":"baseline","scale_factor":1})
    assert missing.status_code==404
    assert invalid.status_code==422

def test_processing_service_rejects_oversized_baseline_before_creating_output(monkeypatch):
    from app import services

    monkeypatch.setattr(settings,"max_baseline_output_pixels",15)
    source=settings.data_dir/"uploads"/"service-input.tif"
    with rasterio.open(source,"w",driver="GTiff",width=2,height=2,count=1,dtype="uint16",
                       crs="EPSG:32643",transform=from_origin(500000,2000000,10,10)) as dataset:
        dataset.write(np.ones((1,2,2),dtype="uint16"))
    monkeypatch.setattr(services,"image_path",lambda _image_id:source)
    with pytest.raises(ValueError,match="output exceeds configured pixel limit"):
        services.process("service-input",{"id":"direct-call","method":"baseline","scale_factor":2})
    assert list((settings.data_dir/"outputs").iterdir())==[]

def test_rejected_upload_cleans_partial_raster_and_preview(monkeypatch):
    monkeypatch.setattr(main_module,"ident",lambda:"rejected-upload")
    def write_partial_preview(_src,dest):
        dest.write_bytes(b"partial preview")
    def reject_metadata(_kind,_key,_item):
        raise HTTPException(status_code=413,detail="simulated upload rejection")
    monkeypatch.setattr(main_module,"make_preview",write_partial_preview)
    monkeypatch.setattr(main_module,"save_json",reject_metadata)
    response=upload_raster()
    assert response.status_code==413
    assert not (settings.data_dir/"uploads"/"rejected-upload.tif").exists()
    assert not (settings.data_dir/"previews"/"rejected-upload.png").exists()

def test_invalid_parameters_and_metrics(tmp_path):
    from app.services import compare_metrics
    up=client.post('/api/imagery/upload',files={'file':('metrics.tif',raster_bytes(),'image/tiff')})
    image=up.json()
    invalid=client.post('/api/processing/start',json={'image_id':image['id'],'scale_factor':9})
    assert invalid.status_code==422
    ref=tmp_path/'ref.tif'
    with rasterio.open(ref,'w',driver='GTiff',width=8,height=6,count=3,dtype='uint16',crs='EPSG:32643',transform=from_origin(500000,2000000,10,10)) as ds:
        ds.write(np.ones((3,6,8),dtype='uint16')*100)
    assert compare_metrics(str(ref),str(ref))=={'rmse':0.0,'psnr':float('inf'),'ssim':1.0}

def test_chunked_metrics_match_reference_for_identical_images(tmp_path,monkeypatch):
    from app.services import compare_metrics

    monkeypatch.setattr(settings,"metric_chunk_size",5)
    data=np.arange(3*13*17,dtype=np.float32).reshape(3,13,17)
    pred=write_metric_raster(tmp_path/"identical-pred.tif",data)
    ref=write_metric_raster(tmp_path/"identical-ref.tif",data)
    result=compare_metrics(str(pred),str(ref))
    assert result=={"rmse":0.0,"psnr":float("inf"),"ssim":1.0}
    assert_metrics_match(result,full_array_metric_reference(pred,ref))

def test_chunked_metrics_match_reference_for_known_differences_and_edge_chunks(tmp_path,monkeypatch):
    from app.services import compare_metrics

    monkeypatch.setattr(settings,"metric_chunk_size",5)
    ref_data=np.arange(3*13*17,dtype=np.float32).reshape(3,13,17)
    pred_data=ref_data.copy()
    pred_data[0,2,3]+=9
    pred_data[2,-1,-1]-=2
    pred=write_metric_raster(tmp_path/"different-pred.tif",pred_data)
    ref=write_metric_raster(tmp_path/"different-ref.tif",ref_data)
    assert_metrics_match(compare_metrics(str(pred),str(ref)),full_array_metric_reference(pred,ref))

def test_chunked_metrics_preserve_nodata_behavior(tmp_path,monkeypatch):
    from app.services import compare_metrics

    monkeypatch.setattr(settings,"metric_chunk_size",4)
    ref_data=np.arange(2*11*13,dtype=np.float32).reshape(2,11,13)
    pred_data=ref_data.copy()
    ref_data[0,1,1]=np.nan
    pred_data[1,8,9]=np.nan
    pred_data[0,4,6]+=3
    pred=write_metric_raster(tmp_path/"nodata-pred.tif",pred_data,nodata=np.nan)
    ref=write_metric_raster(tmp_path/"nodata-ref.tif",ref_data,nodata=np.nan)
    assert_metrics_match(compare_metrics(str(pred),str(ref)),full_array_metric_reference(pred,ref))

def test_no_valid_pixels_error_precedes_small_image_ssim_error(tmp_path):
    from app.services import compare_metrics

    invalid=np.full((1,2,2),np.nan,dtype=np.float32)
    pred=write_metric_raster(tmp_path/"all-nodata-pred.tif",invalid,nodata=np.nan)
    ref=write_metric_raster(tmp_path/"all-nodata-ref.tif",invalid,nodata=np.nan)
    with pytest.raises(ValueError,match="No valid overlapping pixels"):
        compare_metrics(str(pred),str(ref))

def test_too_small_image_keeps_existing_ssim_error(tmp_path):
    from app.services import compare_metrics

    data=np.ones((1,2,2),dtype=np.float32)
    pred=write_metric_raster(tmp_path/"small-pred.tif",data)
    ref=write_metric_raster(tmp_path/"small-ref.tif",data)
    with pytest.raises(ValueError,match="Images are too small for reliable SSIM"):
        compare_metrics(str(pred),str(ref))

@pytest.mark.parametrize("mismatch",["dimensions","crs","transform","bands"])
def test_chunked_metrics_keep_mismatch_errors(tmp_path,mismatch):
    from app.services import compare_metrics

    base=np.ones((3,9,11),dtype=np.float32)
    pred=write_metric_raster(tmp_path/"mismatch-pred.tif",base)
    if mismatch=="dimensions":
        ref=write_metric_raster(tmp_path/"mismatch-ref.tif",base[:,:-1,:])
    elif mismatch=="crs":
        ref=write_metric_raster(tmp_path/"mismatch-ref.tif",base,crs="EPSG:4326")
    elif mismatch=="transform":
        ref=write_metric_raster(tmp_path/"mismatch-ref.tif",base,
                                transform=from_origin(500001,2000000,10,10))
    else:
        ref=write_metric_raster(tmp_path/"mismatch-ref.tif",base[:2])
    with pytest.raises(ValueError,match="Rasters must have matching CRS, transform, dimensions, and band count"):
        compare_metrics(str(pred),str(ref))

def test_chunked_metric_reads_are_bounded_windows(tmp_path,monkeypatch):
    from app import services

    monkeypatch.setattr(settings,"metric_chunk_size",5)
    data=np.arange(2*13*17,dtype=np.float32).reshape(2,13,17)
    pred=write_metric_raster(tmp_path/"tracked-pred.tif",data)
    ref=write_metric_raster(tmp_path/"tracked-ref.tif",data+1)
    original_open=rasterio.open
    reads=[]

    class TrackingDataset:
        def __init__(self,dataset):
            self.dataset=dataset
        def __enter__(self):
            self.dataset.__enter__()
            return self
        def __exit__(self,*args):
            return self.dataset.__exit__(*args)
        def __getattr__(self,name):
            return getattr(self.dataset,name)
        def read(self,*args,**kwargs):
            window=kwargs.get("window")
            assert window is not None
            reads.append((int(window.width),int(window.height)))
            return self.dataset.read(*args,**kwargs)

    monkeypatch.setattr(services.rasterio,"open",lambda *args,**kwargs:TrackingDataset(original_open(*args,**kwargs)))
    services.compare_metrics(str(pred),str(ref))
    assert reads
    assert max(max(width,height) for width,height in reads)<=settings.metric_chunk_size+6

def test_metric_chunk_size_requires_positive_value():
    with pytest.raises(ValidationError):
        Settings.model_validate({"metric_chunk_size":0})
    with pytest.raises(ValidationError):
        Settings.model_validate({"metric_chunk_size":5.5})

def test_metric_api_response_fields_remain_unchanged():
    from app.services import save_json

    uploaded=upload_raster().json()
    reference_path=settings.data_dir/"uploads"/uploaded["stored_name"]
    output_path=settings.data_dir/"outputs"/"metric-api-output.tif"
    shutil.copyfile(reference_path,output_path)
    with rasterio.open(output_path,"r+") as output:
        first_band=output.read(1)
        first_band[2,2]+=5
        output.write(first_band,1)
    save_json("job","metric-api-job",{"status":"completed","result":{"output_path":str(output_path)}})
    response=client.post("/api/validation/compare",params={
        "output_job_id":"metric-api-job","reference_image_id":uploaded["id"]})
    assert response.status_code==200,response.text
    assert response.json()["available"] is True
    assert set(response.json()["metrics"])=={"rmse","psnr","ssim"}

def test_missing_trained_weights_fails_cleanly():
    up=client.post('/api/imagery/upload',files={'file':('trained.tif',raster_bytes(),'image/tiff')})
    response=client.post('/api/processing/start',json={'image_id':up.json()['id'],'method':'trained_srcnn','scale_factor':2})
    assert response.status_code==202
    job=client.get('/api/jobs/'+response.json()['id']).json()
    assert job['status']=='failed'
    assert 'checkpoint' in job['error'].lower() or 'pytorch' in job['error'].lower()
