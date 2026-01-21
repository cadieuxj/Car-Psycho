export default function Home() {
  return (
    <main style={{ padding: '2rem', fontFamily: 'system-ui' }}>
      <h1>Car-Psycho</h1>
      <p>Psychometric Car Sales Platform</p>
      <p style={{ color: '#666', marginTop: '1rem' }}>
        Frontend under development...
      </p>
      <div style={{ marginTop: '2rem' }}>
        <h2>Services Status</h2>
        <ul>
          <li>Manager Service: <a href="http://localhost:8001" target="_blank">Port 8001</a></li>
          <li>Inference Service: <a href="http://localhost:8002" target="_blank">Port 8002</a></li>
          <li>Data Service: <a href="http://localhost:8003" target="_blank">Port 8003</a></li>
          <li>API Gateway: <a href="http://localhost:8080" target="_blank">Port 8080</a></li>
        </ul>
      </div>
    </main>
  )
}
