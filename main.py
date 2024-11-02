import os
from resume_parser import iface  # Import Gradio interface from resume_parser
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8000)))