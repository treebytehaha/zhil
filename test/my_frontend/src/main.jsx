import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./style.css"; // 自定义增强样式（滚动条、代码块等）

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
