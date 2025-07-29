import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { FiUpload, FiSend, FiFile, FiLoader, FiAlertCircle } from 'react-icons/fi';

export default function DataAgentApp() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('选择文件');
  const [command, setCommand] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('请先选择文件');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append('file', file);
      
      await axios.post('/api/upload-excel', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setResponse({
        type: 'success',
        message: `${fileName} 上传成功！现在可以输入分析命令。`,
        data: null
      });
    } catch (err) {
      setError(`上传失败: ${err.response?.data?.message || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCommandSubmit = async () => {
    if (!command.trim()) {
      setError('请输入分析命令');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      const res = await axios.post('/api/analyze', { command });
      
      setResponse({
        type: 'analysis',
        message: '分析结果',
        data: res.data.result
      });
    } catch (err) {
      setError(`分析失败: ${err.response?.data?.message || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-6"
    >
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
        {/* 头部 */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-6 text-white">
          <h1 className="text-2xl font-bold flex items-center">
            <FiFile className="mr-2" />
            数据统计前端 (LangChain Agent)
          </h1>
          <p className="text-blue-100 mt-1">上传Excel文件并使用自然语言命令分析数据</p>
        </div>
        
        {/* 主体内容 */}
        <div className="p-6 grid gap-6">
          {/* 文件上传区域 */}
          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-gray-700">1. 上传Excel文件</h2>
            <div className="flex items-center gap-3">
              <label className="flex-1">
                <input 
                  type="file" 
                  accept=".xlsx,.xls" 
                  onChange={handleFileChange}
                  className="hidden"
                  id="fileInput"
                />
                <div className="flex items-center justify-between p-3 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-500 transition-colors">
                  <span className="text-gray-600 truncate max-w-xs">
                    {fileName}
                  </span>
                  <FiUpload className="text-gray-500" />
                </div>
              </label>
              <button 
                onClick={handleUpload} 
                disabled={loading || !file}
                className={`flex items-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors ${
                  loading || !file 
                    ? 'bg-gray-200 text-gray-500 cursor-not-allowed' 
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {loading ? <FiLoader className="animate-spin" /> : <FiUpload />}
                上传
              </button>
            </div>
          </div>
          
          {/* 命令输入区域 */}
          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-gray-700">2. 输入分析命令</h2>
            <div className="relative">
              <input
                type="text"
                placeholder="例如：统计2024年2月PIX出现的次数"
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                className="w-full p-4 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                onKeyPress={(e) => e.key === 'Enter' && handleCommandSubmit()}
              />
              <button 
                onClick={handleCommandSubmit}
                disabled={loading || !command.trim()}
                className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-full transition-colors ${
                  loading || !command.trim()
                    ? 'text-gray-400 cursor-not-allowed'
                    : 'text-blue-600 hover:bg-blue-50'
                }`}
              >
                <FiSend />
              </button>
            </div>
            <div className="text-sm text-gray-500 pl-2">
              示例命令: "统计销售额最高的5个产品", "比较2023年和2024年的销售趋势"
            </div>
          </div>
          
          {/* 错误提示 */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700"
            >
              <FiAlertCircle className="mt-0.5 flex-shrink-0" />
              <div>{error}</div>
            </motion.div>
          )}
          
          {/* 结果展示 */}
          {(response || loading) && (
            <div className="mt-4">
              <h2 className="text-lg font-semibold text-gray-700 mb-3">分析结果</h2>
              {loading ? (
                <div className="flex items-center justify-center p-8 text-gray-500">
                  <FiLoader className="animate-spin mr-2" />
                  正在分析数据，请稍候...
                </div>
              ) : (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="bg-gray-50 rounded-xl p-4 border border-gray-200 overflow-hidden"
                >
                  {response.type === 'success' ? (
                    <div className="text-green-700">{response.message}</div>
                  ) : (
                    <>
                      <div className="font-medium text-gray-700 mb-2">{response.message}</div>
                      <pre className="whitespace-pre-wrap break-words bg-white p-3 rounded-lg border border-gray-200 text-sm overflow-x-auto">
                        {typeof response.data === 'string' 
                          ? response.data 
                          : JSON.stringify(response.data, null, 2)}
                      </pre>
                    </>
                  )}
                </motion.div>
              )}
            </div>
          )}
        </div>
        
        {/* 页脚 */}
        <div className="bg-gray-50 px-6 py-3 text-center text-sm text-gray-500 border-t border-gray-200">
          数据统计系统 &copy; {new Date().getFullYear()}
        </div>
      </div>
    </motion.div>
  );
}