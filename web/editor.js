export const DEFAULT_GRID=['#########','#S..#..B#','#...#...#','#...D...#','#...#...#','#K..#...#','#...#...#','#E......#','#########'];
const tools=[['#','Wall'],['.','Floor'],['S','Joyce'],['E','Exit'],['B','Banana'],['K','Key'],['D','Door'],['C','Camera']];
const symbols={'#':'','.' :'',S:'●',E:'↗',B:'◡',K:'⚿',D:'▥',C:'◉'};
export class MapEditor{
 constructor(board,palette){this.board=board;this.palette=palette;this.grid=DEFAULT_GRID.map(r=>r.split(''));this.tool='#';this.buttons=[];for(const [tile,label] of tools){const b=document.createElement('button');b.textContent=label;b.dataset.tool=tile;b.classList.toggle('active',tile===this.tool);b.setAttribute('aria-pressed',String(tile===this.tool));b.addEventListener('click',()=>{this.tool=tile;for(const p of this.palette.children){p.classList.toggle('active',p.dataset.tool===tile);p.setAttribute('aria-pressed',String(p.dataset.tool===tile));}});palette.append(b);}for(let y=0;y<9;y++)for(let x=0;x<9;x++){const b=document.createElement('button');b.addEventListener('click',()=>this.paint(x,y));board.append(b);this.buttons.push(b);}this.render();}
 paint(x,y){if((x===0||y===0||x===8||y===8)&&this.tool!=='#')return;if(['S','E','B','K','D','C'].includes(this.tool))for(let r=0;r<9;r++)for(let c=0;c<9;c++)if(this.grid[r][c]===this.tool)this.grid[r][c]='.';this.grid[y][x]=this.tool;this.render();}
 render(){for(let y=0;y<9;y++)for(let x=0;x<9;x++){const b=this.buttons[y*9+x],tile=this.grid[y][x];b.dataset.tile=tile;b.textContent=symbols[tile];b.setAttribute('aria-label',`${tools.find(t=>t[0]===tile)?.[1]||tile}, column ${x+1}, row ${y+1}`);}}
 reset(){this.grid=DEFAULT_GRID.map(r=>r.split(''));this.render();}
 get value(){return this.grid.map(r=>r.join(''));}
}
