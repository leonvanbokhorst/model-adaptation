
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
import subprocess

torch.manual_seed(42)

embed_dim = 32
num_agents = 10

def interpret_embedding_with_llm(embedding):
    emb_list = embedding.squeeze().tolist()
    emb_str = ", ".join([f"{x:.2f}" for x in emb_list])
    
    prompt = f"""
Denk aan deze vector van innerlijke toestanden: [{emb_str}].
Beschrijf deze vector alsof het een innerlijk landschap is met emoties, doelen, trauma's en verlangens.
Geef een semantische interpretatie: welke gevoelens, politieke overtuigingen, of verlangens suggereren deze waarden?
"""

    # Debug print
    print("\n[DEBUG] Calling Ollama with prompt:")
    print(prompt)
    
    result = subprocess.run(
        ["ollama", "run", "hermes3:latest"], 
        input=prompt.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    output_text = result.stdout.decode('utf-8').strip()
    print("[DEBUG] Ollama output:", output_text)
    
    if len(output_text) > 0:
        ascii_vals = [ord(c) for c in output_text if ord(c) < 128]
        if ascii_vals:
            mean_ascii = sum(ascii_vals) / len(ascii_vals)
        else:
            mean_ascii = 0.0
    else:
        mean_ascii = 0.0
    
    interpreted_embedding = embedding + (mean_ascii % 100)*1e-4
    return interpreted_embedding


class Agent(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        
        self.personality = nn.Parameter(torch.randn(embed_dim) * 0.5, requires_grad=False)
        self.trauma = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)
        self.emotion = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)
        self.politics = nn.Parameter(torch.randn(embed_dim)*0.1, requires_grad=False)
        self.desires = nn.Parameter(torch.randn(embed_dim)*0.1, requires_grad=False)
        
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim*4 + embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, embed_dim)
        )
        
        self.external_integration = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
    
    def forward(self, external_context):
        internal_states = torch.cat([self.personality, 
                                     self.emotion, 
                                     self.trauma,
                                     self.politics,
                                     self.desires])
        combined = self.combiner(internal_states.unsqueeze(0))
        integrated = combined + self.external_integration(external_context)
        
        interpreted = interpret_embedding_with_llm(integrated)
        
        with torch.no_grad():
            self.emotion += 0.001 * (interpreted.squeeze(0) - self.emotion)
            self.trauma *= 0.999
            self.desires += 0.0005 * (interpreted.squeeze(0) - self.desires)

        return interpreted.squeeze(0)


agents = [Agent(embed_dim) for _ in range(num_agents)]

def get_external_context(step):
    angle = torch.tensor(step * 0.01)
    context = torch.zeros(embed_dim)
    context[0] = torch.sin(angle) * 0.5
    context[1] = torch.cos(angle) * 0.5
    context += 0.01 * torch.randn(embed_dim)
    return context.unsqueeze(0)

plt.ion()
fig, ax = plt.subplots()

def visualize(embeddings, step):
    ax.clear()
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(embeddings.detach().numpy())
    ax.scatter(points_2d[:,0], points_2d[:,1], c='blue', alpha=0.7)
    ax.set_title(f"Step {step}: Agent States in 2D Projection")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.draw()
    plt.pause(0.01)
    print(f"[DEBUG] Visualization updated at step {step}")


step = 0
try:
    while True:
        step += 1
        external_context = get_external_context(step)
        
        print(f"\n[DEBUG] Step {step} start, external_context mean: {external_context.mean().item():.4f}")
        
        outputs = []
        for i, agent in enumerate(agents):
            out = agent(external_context)
            outputs.append(out)
            print(f"[DEBUG] Agent {i} output mean: {out.mean().item():.4f}")
        
        outputs = torch.stack(outputs)
        mean_embedding = outputs.mean(dim=0)
        print(f"[DEBUG] mean_embedding mean: {mean_embedding.mean().item():.4f}")
        
        direction_factor = 1.0 if mean_embedding.mean() < 0 else -1.0
        print(f"[DEBUG] direction_factor: {direction_factor}")
        
        for i, agent in enumerate(agents):
            with torch.no_grad():
                agent.desires += 0.0001 * direction_factor * (agent.desires - mean_embedding)
                if torch.rand(1).item() < 0.001:
                    agent.trauma += 0.1 * torch.randn(embed_dim)
                    print(f"[DEBUG] Agent {i} trauma triggered.")
        
        if step % 50 == 0:
            visualize(outputs, step)
        
        time.sleep(0.5)  # iets langer wachten om updates te kunnen zien

except KeyboardInterrupt:
    print("Simulatie gestopt door gebruiker")
    plt.show()  # laat het laatste plot staan
