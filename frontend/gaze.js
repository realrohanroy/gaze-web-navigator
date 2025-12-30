const cursor = document.getElementById("gaze-cursor");

const socket = new WebSocket("ws://localhost:8765");

socket.onopen = () => {
  console.log("WebSocket connected");
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  
  const x = data.nx * vw;
  const y = data.ny * vh;
  
  cursor.style.left = x + "px";
  cursor.style.top = y + "px";
  
};

socket.onerror = (e) => {
  console.error("WS error", e);
};
