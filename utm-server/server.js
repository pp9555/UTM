const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });

console.log("WebSocket Server running on ws://localhost:8080");

wss.on('connection', function connection(ws) {
  console.log("New client connected!");

  ws.on('message', function incoming(data) {
    // Broadcast incoming data to all connected clients (React)
    wss.clients.forEach(function each(client) {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(data.toString());
      }
    });
  });

  ws.on('close', () => {
    console.log("Client disconnected");
  });
});