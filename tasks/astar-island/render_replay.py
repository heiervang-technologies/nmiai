import json
import glob
from pathlib import Path

def generate_replay_viewer():
    # Use the explicitly requested Seed 1 (seed_index 0) from Round 18
    replay_file = 'tasks/astar-island/replays/round18_seed0_simseed1.json'
    print(f"Generating viewer for {replay_file}...")
    
    with open(replay_file) as f:
        data = json.load(f)
        
    frames = [frame['grid'] for frame in data['frames']]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>50-Year Simulation Replay</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; }}
  .grid {{ display: grid; grid-template-columns: repeat(40, 15px); grid-template-rows: repeat(40, 15px); gap: 1px; background: #333; border: 2px solid #555; margin-top: 20px; }}
  .cell {{ width: 15px; height: 15px; }}
  .c0 {{ background: #bba080; }} /* Plains/Empty fallback */
  .c1 {{ background: #ff4444; }} /* Settlement */
  .c2 {{ background: #aa00ff; }} /* Port */
  .c3 {{ background: #000000; }} /* Ruin */
  .c4 {{ background: #228b22; }} /* Forest */
  .c5 {{ background: #888888; }} /* Mountain */
  .c10 {{ background: #0000aa; }} /* Ocean */
  .c11 {{ background: #bba080; }} /* Plains */
  .controls {{ margin-top: 20px; text-align: center; }}
  input[type=range] {{ width: 400px; }}
  button {{ padding: 10px 20px; margin: 10px; cursor: pointer; background: #444; color: white; border: none; border-radius: 4px; }}
  button:hover {{ background: #666; }}
  .legend {{ margin-top: 20px; display: flex; gap: 15px; flex-wrap: wrap; justify-content: center; max-width: 600px; }}
  .leg-item {{ display: flex; align-items: center; gap: 5px; font-size: 14px; }}
  .leg-box {{ width: 15px; height: 15px; border: 1px solid #555; }}
</style>
</head>
<body>
  <h2>Replay Viewer: {Path(replay_file).name}</h2>
  
  <div class="controls">
    <label>Year / Step: <span id="step-label">0</span> / {len(frames)-1}</label><br>
    <input type="range" id="step-slider" min="0" max="{len(frames)-1}" value="0"><br>
    <button id="play-btn">Play / Pause</button>
  </div>

  <div id="grid" class="grid"></div>

  <div class="legend">
    <div class="leg-item"><div class="leg-box c11"></div> Plains</div>
    <div class="leg-item"><div class="leg-box c1"></div> Settlement</div>
    <div class="leg-item"><div class="leg-box c2"></div> Port</div>
    <div class="leg-item"><div class="leg-box c3"></div> Ruin</div>
    <div class="leg-item"><div class="leg-box c4"></div> Forest</div>
    <div class="leg-item"><div class="leg-box c5"></div> Mountain</div>
    <div class="leg-item"><div class="leg-box c10"></div> Ocean</div>
  </div>

  <script>
    const frames = {json.dumps(frames)};
    const gridEl = document.getElementById('grid');
    const slider = document.getElementById('step-slider');
    const label = document.getElementById('step-label');
    const playBtn = document.getElementById('play-btn');
    
    // Initialize grid DOM elements
    for (let y = 0; y < 40; y++) {{
      for (let x = 0; x < 40; x++) {{
        const cell = document.createElement('div');
        cell.id = `cell-${{y}}-${{x}}`;
        cell.className = 'cell';
        gridEl.appendChild(cell);
      }}
    }}

    function renderFrame(step) {{
      label.innerText = step;
      slider.value = step;
      const gridData = frames[step];
      for (let y = 0; y < 40; y++) {{
        for (let x = 0; x < 40; x++) {{
          const cell = document.getElementById(`cell-${{y}}-${{x}}`);
          let val = gridData[y][x];
          cell.className = 'cell c' + val;
        }}
      }}
    }}

    slider.addEventListener('input', (e) => {{
      renderFrame(parseInt(e.target.value));
    }});

    let playing = false;
    let playInterval;

    playBtn.addEventListener('click', () => {{
      playing = !playing;
      if (playing) {{
        if (parseInt(slider.value) >= frames.length - 1) slider.value = 0;
        playInterval = setInterval(() => {{
          let step = parseInt(slider.value) + 1;
          if (step >= frames.length) {{
            clearInterval(playInterval);
            playing = false;
          }} else {{
            renderFrame(step);
          }}
        }}, 200); // 5 FPS
      }} else {{
        clearInterval(playInterval);
      }}
    }});

    // Render initial frame
    renderFrame(0);
  </script>
</body>
</html>"""

    out_path = 'tasks/astar-island/view_50_step_replay.html'
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Viewer created at {out_path}")

if __name__ == "__main__":
    generate_replay_viewer()