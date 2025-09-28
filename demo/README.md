# Industrial Audio RAG Demo

Interactive HuggingFace Spaces demo for the Industrial Audio RAG system.

## 🎯 Demo Features

- **Interactive Chat Interface**: Ask questions about industrial audio data
- **Real-time Performance Metrics**: See embedding, search, and LLM response times
- **Source Attribution**: View relevant audio clips and metadata used for answers
- **Sample Questions**: Pre-loaded examples to get started quickly
- **System Information**: Live status and configuration details

## 🚀 Local Development

### Setup

```bash
# Navigate to demo directory
cd demo

# Install dependencies
pip install -r requirements.txt

# Set API endpoint (optional)
export API_BASE_URL="http://localhost:8000"
export API_KEY="your-api-key"  # if authentication enabled

# Run demo
python app.py
```

### Configuration

Environment variables:

- `API_BASE_URL`: Backend API endpoint (default: demo server)
- `API_KEY`: API authentication key (optional)

## 🌐 HuggingFace Spaces Deployment

### Files Structure

```
demo/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── spaces_config.yml  # HF Spaces configuration
```

### Deployment Steps

1. **Create HuggingFace Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose "Gradio" as the SDK
   - Set space name: `industrial-audio-rag`

2. **Upload Files**:
   ```bash
   # Clone your space repository
   git clone https://huggingface.co/spaces/yourusername/industrial-audio-rag
   cd industrial-audio-rag
   
   # Copy demo files
   cp /path/to/demo/* .
   
   # Push to HuggingFace
   git add .
   git commit -m "Initial demo deployment"
   git push
   ```

3. **Configure Environment**:
   - In HuggingFace Spaces settings, add secrets:
     - `API_BASE_URL`: Your deployed API endpoint
     - `API_KEY`: API authentication key (if needed)

## 🎨 Demo Interface

### Main Components

1. **Query Input**: Natural language questions about audio data
2. **Configuration**: Adjust number of results to retrieve
3. **Sample Questions**: Pre-loaded examples covering common use cases
4. **System Information**: Live API status and model details

### Output Sections

1. **Answer**: AI-generated response based on retrieved context
2. **Performance Metrics**: Real-time timing and performance data
3. **Sources**: Relevant audio clips and metadata used for the answer

### Sample Questions

- "Which bearing clips in section 00 show dominant frequency above 900 Hz?"
- "Find all clips with bearing anomalies in the training dataset"
- "What are the frequency characteristics of normal vs abnormal bearings?"
- "Show me clips with high energy content in the 1-2 kHz range"

## 🔧 Customization

### Adding New Sample Questions

Edit the `SAMPLE_QUERIES` list in `app.py`:

```python
SAMPLE_QUERIES = [
    "Your new sample question here",
    # ... existing questions
]
```

### Styling

Modify the CSS in the `gr.Blocks` configuration:

```python
css="""
.your-custom-class { 
    /* Your styles here */ 
}
"""
```

### API Integration

The demo connects to your deployed API via:

- `/ask` endpoint for queries
- `/info` endpoint for system information
- Optional authentication via Bearer tokens

## 📊 Performance Features

### Real-time Metrics

The demo displays:
- Total response time
- Embedding generation time
- Vector search time
- LLM generation time
- Number of sources retrieved

### Error Handling

- Graceful handling of API timeouts
- User-friendly error messages
- Fallback responses for demo continuity

## 🚀 Production Deployment

### Environment Configuration

For production deployment:

```bash
# Set production API endpoint
export API_BASE_URL="https://your-production-api.com"

# Enable authentication
export API_KEY="your-production-api-key"
```

### Scaling Considerations

- The demo is stateless and can handle multiple concurrent users
- API rate limiting is handled gracefully
- Automatic retry logic for transient failures

## 🎥 Demo Flow

1. **Welcome**: User sees introduction and sample questions
2. **Query**: User enters question or clicks sample
3. **Processing**: Real-time progress indicator during API call
4. **Results**: Answer, metrics, and sources displayed
5. **Exploration**: User can try different questions or adjust parameters

## 📱 Mobile Support

The Gradio interface is mobile-responsive and works well on:
- Desktop browsers
- Tablet devices
- Mobile phones (with responsive layout)

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Check `API_BASE_URL` configuration
   - Verify API is running and accessible
   - Check authentication if enabled

2. **Slow Response Times**:
   - API server may be cold-starting
   - Large queries require more processing time
   - Check network connectivity

3. **Missing Results**:
   - Query may be too specific
   - Try broader or different terminology
   - Check if API has data loaded

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This demo provides an engaging way to showcase the Industrial Audio RAG system's capabilities to potential users, employers, and collaborators.