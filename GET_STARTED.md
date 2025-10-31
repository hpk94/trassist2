# 🚀 Get Started with LiteLLM

Your code has been refactored! Here's everything you need to know in one page.

---

## ⚡ Quick Install (30 seconds)

```bash
# 1. Install new dependency
pip install -r requirements.txt

# 2. Add to .env file
echo "LITELLM_MODEL=gpt-4o" >> .env

# 3. Run!
python app.py
```

**That's it!** Your code now works with multiple AI providers.

---

## 🎯 What Changed?

### Before
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="chatgpt-4o-latest", ...)
```

### After
```python
import litellm
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o")
response = litellm.completion(model=LITELLM_MODEL, ...)
```

**Everything else stays the same!** Same functionality, more flexibility.

---

## 🔧 Configuration

### Your `.env` file should have:

```bash
# Choose your AI model
LITELLM_MODEL=gpt-4o

# Add the matching API key
OPENAI_API_KEY=sk-your-key-here

# Your existing MEXC config (unchanged)
MEXC_API_KEY=...
MEXC_API_SECRET=...
```

---

## 🤖 Available Models

### For Chart Analysis (Vision Required)

| Model | Command | Cost | Speed |
|-------|---------|------|-------|
| **GPT-4o** (default) | `gpt-4o` | $$ | Fast ⚡ |
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20241022` | $$$ | Medium |
| **Gemini 1.5 Pro** | `gemini/gemini-1.5-pro` | $ | Fast ⚡ |

### Switch Models

**Option 1:** Edit `.env` file
```bash
LITELLM_MODEL=claude-3-5-sonnet-20241022
```

**Option 2:** Command line
```bash
LITELLM_MODEL=gemini/gemini-1.5-pro python app.py
```

---

## 🆘 Troubleshooting

### Error: "litellm could not be resolved"
```bash
pip install litellm
```

### Error: "AuthenticationError"
Check your API key matches your model:
- `gpt-4o` needs `OPENAI_API_KEY`
- `claude-*` needs `ANTHROPIC_API_KEY`
- `gemini/*` needs `GEMINI_API_KEY`

### Error: "Model not found"
Check the model name:
- ✅ `gpt-4o` (correct)
- ❌ `gpt4o` (wrong)
- ✅ `gemini/gemini-1.5-pro` (needs prefix)

---

## 💡 Want to Save 50%? Use DeepSeek!

Use DeepSeek for trade gate decisions and save big:

```bash
# In .env
LITELLM_VISION_MODEL=gpt-4o              # For charts
LITELLM_TEXT_MODEL=deepseek/deepseek-chat # For trade gates
DEEPSEEK_API_KEY=sk-your-key
```

**See:** [`DEEPSEEK_QUICKSTART.md`](DEEPSEEK_QUICKSTART.md) for 30-second setup!

## 📚 Documentation Map

```
GET_STARTED.md         ← You are here! Quick overview
│
├─ DEEPSEEK_QUICKSTART.md  ← 30-sec DeepSeek setup (save 50%!)
├─ DEEPSEEK_GUIDE.md   ← Complete DeepSeek guide
├─ QUICKSTART.md       ← 2-minute setup guide
├─ CHANGES.md          ← Complete changelog
├─ MIGRATION_GUIDE.md  ← Detailed migration steps
├─ LITELLM_CONFIGURATION.md  ← All models & config
├─ REFACTOR_SUMMARY.md ← Technical details
└─ README.md           ← Project overview
```

---

## ✅ Checklist

Before running in production:

- [ ] Installed litellm: `pip install -r requirements.txt`
- [ ] Set `LITELLM_MODEL` in `.env`
- [ ] Set appropriate API key in `.env`
- [ ] Tested with: `python app.py`
- [ ] Verified chart analysis works
- [ ] Checked JSON output in `llm_outputs/`
- [ ] Tested web app (optional): `python web_app.py`

---

## 💡 Pro Tips

### 1. Test Multiple Models
```bash
# Compare results
LITELLM_MODEL=gpt-4o python app.py
LITELLM_MODEL=claude-3-5-sonnet-20241022 python app.py
LITELLM_MODEL=gemini/gemini-1.5-pro python app.py

# Check llm_outputs/ to compare
```

### 2. Save Money
```bash
# Development/testing: use cheaper model
LITELLM_MODEL=gemini/gemini-1.5-pro

# Production: use best model
LITELLM_MODEL=gpt-4o
```

### 3. Local Development
```bash
# Use Ollama for free local testing
LITELLM_MODEL=ollama/llama2
```

---

## 🎉 Benefits

✅ **No vendor lock-in** - Switch providers anytime  
✅ **Cost optimization** - Choose based on budget  
✅ **Better reliability** - Fallback to other models if one fails  
✅ **Future-proof** - New models supported automatically  
✅ **Same code** - No changes needed to switch models  

---

## 📞 Need Help?

1. **Quick answers**: Read [`QUICKSTART.md`](QUICKSTART.md)
2. **All models**: Check [`LITELLM_CONFIGURATION.md`](LITELLM_CONFIGURATION.md)
3. **Step-by-step**: Follow [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)
4. **Technical details**: Review [`REFACTOR_SUMMARY.md`](REFACTOR_SUMMARY.md)

---

## 🚦 Ready to Go!

Your code is refactored and ready. Start with these commands:

```bash
# Install
pip install -r requirements.txt

# Configure
cat >> .env << EOF
LITELLM_MODEL=gpt-4o
EOF

# Run
python app.py
```

**Everything should work exactly as before, but now you have the power to switch AI models instantly!** 🎊

---

## 📊 Model Comparison Quick Reference

| Model | Provider | Cost/1M | Vision | Quality | Speed |
|-------|----------|---------|--------|---------|-------|
| gpt-4o | OpenAI | ~$6 | ✅ | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ |
| claude-3-5-sonnet | Anthropic | ~$9 | ✅ | ⭐⭐⭐⭐⭐ | ⚡⚡ |
| gemini-1.5-pro | Google | ~$3 | ✅ | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| gpt-4-turbo | OpenAI | ~$20 | ✅ | ⭐⭐⭐⭐ | ⚡⚡ |
| claude-3-opus | Anthropic | ~$45 | ✅ | ⭐⭐⭐⭐⭐ | ⚡ |

---

**Happy Trading! 📈**

