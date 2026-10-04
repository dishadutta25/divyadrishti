(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const API = { status: '/api/status', tracks: '/api/tracks', events: '/api/events', safety: '/api/public-safety', quality: '/api/quality/status', logs: '/api/logs', stream: '/video.mjpg' };
  const video = $('video');
  let streamTimer;
  const value = (obj, ...keys) => { for (const k of keys) if (obj?.[k] !== undefined && obj[k] !== null) return obj[k]; return null; };
  const listFrom = data => Array.isArray(data) ? data : (data?.items || data?.tracks || data?.events || data?.detections || []);
  const fmt = (v, digits=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '—';
  const safe = v => String(v ?? '—');
  async function get(path) {
    const response = await fetch(path, { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
    return response.json();
  }
  function renderStatus(s) {
    const cam = value(s, 'camera_status', 'cameraState', 'camera');
    const cameraOnline = typeof cam === 'object' ? !!value(cam, 'online', 'connected', 'available') : ['online','connected','ready'].includes(String(cam).toLowerCase());
    $('camera-status').textContent = cameraOnline ? 'Online' : (cam ? 'Unavailable' : 'Unknown');
    $('camera-dot').className = `status-dot ${cameraOnline ? 'online' : 'offline'}`;
    $('camera-detail').textContent = typeof cam === 'object' ? safe(value(cam,'name','source','message') || (cameraOnline ? 'Frame source connected' : 'No frame source')) : (safe(value(s,'camera_message','camera_detail') || (cameraOnline ? 'Frame source connected' : 'Camera unavailable')));
    $('fps').textContent = fmt(value(s,'fps','processing_fps'),1);
    $('latency').textContent = fmt(value(s,'inference_time_ms','latency_ms','processing_latency_ms','latency'),0);
    $('device').textContent = safe(value(s,'device') || 'CPU');
    $('device-detail').textContent = value(s,'gpu_active') ? 'CUDA inference verified' : (value(s,'gpu_available') ? 'GPU found · current detector uses CPU' : 'CUDA unavailable · CPU active');
    $('active-count').textContent = safe(value(s,'active_tracks','track_count') ?? '—');
    $('detection-count').textContent = safe(value(s,'detection_count','detections_count') ?? '—');
    $('lost-count').textContent = safe(value(s,'lost_tracks','lost_persons') ?? 0);
    $('warning-count').textContent = safe(value(s,'warnings') ?? 0);
    $('critical-count').textContent = safe(value(s,'critical') ?? 0);
    const mode = String(value(s,'mode','application_mode') || '').toLowerCase();
    document.querySelectorAll('.mode-button').forEach(b => b.classList.toggle('active', mode && ((mode.includes('industrial') && b.dataset.mode==='industrial') || (!mode.includes('industrial') && b.dataset.mode==='human'))));
    const width=value(s,'frame_width','width'), height=value(s,'frame_height','height');
    if(width && height) $('stream-resolution').textContent=`${width} × ${height}`;
    $('stream-caption').textContent = cameraOnline ? 'Live feed with backend overlays' : 'Camera unavailable';
    $('map-caption').textContent = value(s,'localization_status','calibration_status') || 'Calibrated camera plane';
    const banner=$('connection-banner'); banner.className='connection-banner online'; banner.textContent=`Local service connected${mode ? ` · ${mode.toUpperCase()} MODE` : ''}`;
  }
  function renderTracks(payload) {
    const tracks=listFrom(payload); const tbody=$('tracks'); tbody.replaceChildren();
    $('track-updated').textContent=`Updated ${new Date().toLocaleTimeString()}`;
    if(!tracks.length){tbody.innerHTML='<tr><td colspan="6" class="empty-cell">No active tracks reported</td></tr>';}
    tracks.forEach(t=>{
      const id=value(t,'id','track_id'), cls=value(t,'class_name','class','label'), state=value(t,'state','status')||'TRACKING';
      const x=value(t,'world_x','x'), y=value(t,'world_y','y'), speed=value(t,'speed','speed_mps');
      const isImage=value(t,'coordinate_frame')==='image_pixels';
      const px=value(t,'center_x_px'), py=value(t,'center_y_px');
      const pos=x!==null&&y!==null?`${fmt(x)} ${isImage?'px':'m'}, ${fmt(y)} ${isImage?'px':'m'}`:(px!==null&&py!==null?`${fmt(px,0)} px, ${fmt(py,0)} px`:'—');
      const direction=value(t,'direction','heading');
      const tr=document.createElement('tr'); tr.innerHTML=`<td><div class="object-id"></div><div class="object-class"></div></td><td></td><td></td><td></td><td><span class="state-pill"></span></td><td></td>`;
      tr.children[0].querySelector('.object-id').textContent=safe(id); tr.children[0].querySelector('.object-class').textContent=safe(cls);
      const pixelSpeed=value(t,'speed_px_s');
      tr.children[1].textContent=pos; tr.children[2].textContent=speed!==null?`${fmt(speed,1)} m/s${direction?` · ${direction}`:''}`:(pixelSpeed!==null?`${fmt(pixelSpeed,0)} px/s${direction?` · ${direction}`:''}`:'—'); tr.children[3].textContent=safe(value(t,'zone','zone_name'));
      const safety=value(t,'safety_status')||'SAFE';
      const pill=tr.children[4].querySelector('.state-pill'); pill.textContent=`${String(state).toUpperCase()} · ${String(safety).toUpperCase()}`; if(/lost|removed|missing/i.test(state))pill.classList.add('lost'); if(String(safety).toUpperCase()==='WARNING'){pill.style.color='var(--amber)';pill.style.background='#302719';} else if(String(safety).toUpperCase()==='CRITICAL'){pill.style.color='var(--red)';pill.style.background='#301b20';}
      if(['WARNING','CRITICAL','MISSING'].includes(String(safety).toUpperCase())) {
        const action=document.createElement('button');
        action.className='assistance-button';
        action.type='button';
        action.textContent='Request review';
        action.dataset.personId=String(value(t,'id','track_id')||'');
        tr.children[5].append(action);
      } else {
        tr.children[5].textContent='Human verification';
      }
      tbody.append(tr);
    });
    renderMap(tracks);
  }
  function renderMap(tracks) {
    const points=$('map-points'); points.replaceChildren(); let located=0;
    const statusWindow=window.__vxStatus||{}; const imageCoordinates=String(value(statusWindow,'coordinate_unit')||'').toLowerCase().includes('pixel');
    const width=imageCoordinates?(Number(value(statusWindow,'frame_width'))||640):(Number(value(statusWindow,'room_width_m'))||10);
    const height=imageCoordinates?(Number(value(statusWindow,'frame_height'))||480):(Number(value(statusWindow,'room_height_m'))||8);
    tracks.forEach(t=>{const rawX=imageCoordinates?value(t,'center_x_px'):value(t,'world_x','x'), rawY=imageCoordinates?value(t,'center_y_px'):value(t,'world_y','y'); if(rawX===null||rawY===null)return; const x=Number(rawX), y=Number(rawY); if(!Number.isFinite(x)||!Number.isFinite(y))return; located++;
      const dot=document.createElement('div'); dot.className='map-dotpoint'+(/component|object|product|part/i.test(String(value(t,'class_name','class','label')||''))?' component':''); dot.style.left=`${Math.max(2,Math.min(98,x/width*100))}%`; dot.style.top=`${100-Math.max(2,Math.min(98,y/height*100))}%`;
      const label=document.createElement('span'); label.textContent=safe(value(t,'id','track_id')); dot.append(label); points.append(dot);
    }); $('map-empty').hidden=located>0; $('map-unit').textContent=String(value(statusWindow,'coordinate_unit')||'METERS').toUpperCase();
  }
  function renderEvents(payload) {
    const events=listFrom(payload); $('event-count').textContent=String(events.length); const box=$('events'); box.replaceChildren();
    if(!events.length){box.innerHTML='<div class="empty-cell">No events reported</div>';return;}
    events.slice(0,40).forEach(e=>{const row=document.createElement('div');row.className='event-row';const time=value(e,'timestamp','time','created_at');let display='—';if(time){const d=new Date(time);display=Number.isNaN(d.getTime())?String(time):d.toLocaleTimeString();}const level=String(value(e,'severity','level','type')||'info').toLowerCase();row.innerHTML='<div class="event-time"></div><div class="event-mark"></div><div><div class="event-text"></div><div class="event-meta"></div></div>';row.children[0].textContent=display;row.children[1].classList.toggle('warn',/warn|anomaly/i.test(level));row.children[1].classList.toggle('error',/error|critical/i.test(level));row.querySelector('.event-text').textContent=safe(value(e,'message','description','name'));row.querySelector('.event-meta').textContent=[value(e,'object_id','track_id'),value(e,'zone','zone_name'),level.toUpperCase()].filter(Boolean).join(' · ');box.append(row);});
  }
  function renderPublicSafety(s) {
    const bar = $('public-safety');
    const status = String(value(s, 'status') || 'UNKNOWN').toUpperCase();
    const level = ({ SAFE: 'safe', WARNING: 'warning', CRITICAL: 'critical', MISSING: 'missing' })[status] || 'none';
    bar.dataset.level = level;
    $('public-safety-status').textContent = s.person_id ? `${s.person_id} · ${status}` : status;
    $('public-safety-reason').textContent = safe(value(s, 'reason'));
    $('public-safety-action').textContent = `Recommended action: ${safe(value(s, 'recommended_action'))}`;
    const counts = s.counts || {};
    $('safety-warning-count').textContent = safe(counts.WARNING ?? 0);
    $('safety-critical-count').textContent = safe(counts.CRITICAL ?? 0);
    $('safety-missing-count').textContent = safe(counts.MISSING ?? 0);
    $('industrial-alert-count').textContent = safe(value(s,'industrial_alert_count') ?? 0);
    bar.title = `${safe(value(s, 'assessment_scope'))}${s.human_verification_required ? ' · Human verification required' : ''}`;
  }
  function renderQuality(q) {
    const panel=$('quality-panel');
    const status=String(value(q,'status')||'WAITING').toUpperCase();
    const level=status==='CRITICAL'?'critical':status==='WARNING'?'warning':'none';
    panel.dataset.level=level;
    $('quality-status').textContent=q.enabled?`${status} · ${value(q,'severity')==='none'?'No confirmed candidate':`${value(q,'confirmed_candidate_count')||0} confirmed`}`:'Unavailable';
    const candidate=(Array.isArray(q.candidates)&&q.candidates.length)?q.candidates[0]:null;
    const measurement=candidate
      ? ` Latest: ${safe(candidate.inspection_id)} · ${fmt(candidate.length_mm??candidate.length_px,1)} × ${fmt(candidate.width_mm??candidate.width_px,1)} ${safe(value(q,'measurement_unit'))} · contrast ${fmt(candidate.contrast,1)}.`
      : '';
    $('quality-detail').textContent=q.enabled
      ? `${safe(value(q,'assessment_scope'))} ${value(q,'confirmed_candidate_count')||0} confirmed; ${value(q,'pending_candidate_count')||0} awaiting confirmation.${measurement}`
      : 'Visual screening is unavailable. Check quality.yaml and application logs.';
    $('quality-action').textContent=`Operator action: ${safe(value(q,'recommended_action'))}`;
    $('quality-confirmed').textContent=safe(value(q,'confirmed_candidate_count')??0);
  }
  function renderMobileAccess(s) {
    const urls=Array.isArray(s.lan_access_urls)?s.lan_access_urls:[];
    const list=$('mobile-url-list');
    list.replaceChildren();
    if(urls.length) {
      $('mobile-access-label').textContent='Open one of these on the phone';
      urls.forEach(url=>{
        const link=document.createElement('a');
        link.href=url;
        link.textContent=url;
        link.rel='noreferrer';
        list.append(link);
      });
    } else if(location.hostname && !['localhost','127.0.0.1'].includes(location.hostname)) {
      $('mobile-access-label').textContent='This dashboard is already using a network address';
      const link=document.createElement('a');
      link.href=location.origin;
      link.textContent=location.origin;
      list.append(link);
    } else {
      $('mobile-access-label').textContent='No laptop Wi-Fi address detected yet';
      const message=document.createElement('span');
      message.textContent='Check Wi-Fi connection and restart the app.';
      list.append(message);
    }
  }
  function renderLogManifest(logs) {
    document.querySelectorAll('.csv-downloads a').forEach(link=>{
      const name=link.pathname.split('/').pop().replace('.csv','');
      const file=logs[name];
      link.classList.toggle('log-unavailable',!file?.available);
      link.title=file?.available?`${file.bytes} bytes · updated ${file.updated_at}`:'Log file is not available yet';
    });
  }
  async function poll() {
    const results=await Promise.allSettled([get(API.status),get(API.tracks),get(API.events),get(API.safety),get(API.quality),get(API.logs)]);
    if(results[0].status==='fulfilled'){window.__vxStatus=results[0].value;renderStatus(results[0].value);renderMobileAccess(results[0].value);} else {const b=$('connection-banner');b.className='connection-banner error';b.textContent=`Local service unavailable · ${results[0].reason.message}`;$('camera-status').textContent='Unavailable';$('camera-dot').className='status-dot offline';$('camera-detail').textContent='Could not read /api/status';}
    if(results[1].status==='fulfilled')renderTracks(results[1].value); else {$('track-updated').textContent='Track feed unavailable';}
    if(results[2].status==='fulfilled')renderEvents(results[2].value);
    if(results[3].status==='fulfilled')renderPublicSafety(results[3].value);
    if(results[4].status==='fulfilled')renderQuality(results[4].value);
    if(results[5].status==='fulfilled')renderLogManifest(results[5].value);
  }
  document.addEventListener('click', async event => {
    const button=event.target.closest('.assistance-button');
    if(!button)return;
    const personId=button.dataset.personId;
    if(!window.confirm(`Create a local operator-review request for ${personId}? No face image or data will be sent externally.`))return;
    button.disabled=true;
    try {
      const response=await fetch('/api/assistance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({person_id:personId,reason:'operator_observed_concern',operator_confirmed:true})});
      const payload=await response.json();
      if(!response.ok)throw new Error(payload.message||payload.error||`HTTP ${response.status}`);
      $('connection-banner').className='connection-banner online';
      $('connection-banner').textContent=`Local review request saved for ${personId}; follow your approved human handoff process.`;
      await poll();
    } catch(error) {
      $('connection-banner').className='connection-banner error';
      $('connection-banner').textContent=`Could not create local review request · ${error.message}`;
      button.disabled=false;
    }
  });
  video.addEventListener('load',()=>{video.classList.add('loaded');$('video-message').hidden=true;clearTimeout(streamTimer);});
  video.addEventListener('error',()=>{video.classList.remove('loaded');$('video-message').hidden=false;$('video-message').querySelector('strong').textContent='CAMERA UNAVAILABLE';$('video-message').querySelector('small').textContent='No live stream at /video.mjpg';clearTimeout(streamTimer);streamTimer=setTimeout(()=>{video.src=`${API.stream}?t=${Date.now()}`;},5000);});
  $('refresh').addEventListener('click',()=>{poll();video.src=`${API.stream}?t=${Date.now()}`;});
  document.querySelectorAll('.mode-button').forEach(button=>button.addEventListener('click',async()=>{try{await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:button.dataset.mode})});poll();}catch(e){const b=$('connection-banner');b.className='connection-banner error';b.textContent=`Could not change mode · ${e.message}`;}}));
  setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString();},1000); $('clock').textContent=new Date().toLocaleTimeString(); poll();setInterval(poll,1500);
})();
