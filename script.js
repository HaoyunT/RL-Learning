const canvas = document.getElementById("particleCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let particlesArray = [];
const mouse = { x: null, y: null };
const MAX_PARTICLES = 300;
let lastTime = 0;

window.addEventListener("mousemove", (e) => {
  mouse.x = e.x;
  mouse.y = e.y;
  const now = Date.now();
  if (now - lastTime > 16) { // 约60fps
    for (let i = 0; i < 3; i++) {
      particlesArray.push(new Particle());
    }
    lastTime = now;
  }
});

class Particle {
  constructor() {
    this.x = mouse.x;
    this.y = mouse.y;
    this.size = Math.random() * 6 + 1;
    this.speedX = Math.random() * 3 - 1.5;
    this.speedY = Math.random() * 3 - 1.5;
    this.color = `hsl(${Math.random() * 360}, 100%, 50%)`; // 彩色粒子
  }
  update() {
    this.x += this.speedX;
    this.y += this.speedY;
    this.size *= 0.96;
  }
  draw() {
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

function animate() {
  ctx.fillStyle = "rgba(15,17,23,0.2)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  particlesArray.forEach(p => {
    p.update();
    p.draw();
  });

  // 过滤小粒子
  particlesArray = particlesArray.filter(p => p.size > 0.3);

  // 限制粒子总数
  if (particlesArray.length > MAX_PARTICLES) {
    particlesArray.splice(0, particlesArray.length - MAX_PARTICLES);
  }

  requestAnimationFrame(animate);
}
animate();

window.addEventListener('resize', () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});
