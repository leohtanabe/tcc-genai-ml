# CÓDIGO 23b (PROPOSTA, não existe no notebook original): TABELA 2 E RADAR CALCULADOS DOS DADOS
# Substitui o Código 23 (valores fixos). Pré-requisito: df com a coluna `persona` (Código 15).
# Gera (a) a tabela de contagens e médias por persona (Tabela 2 do TCC) e
# (b) o radar com z-scores calculados (média e desvio da base inteira) para os 6 atributos do K-Means.
# STATUS: escrito sem acesso aos dados; precisa ser executado no Colab e conferido.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cols_cluster = [
    'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Perceived_AI_Dependency',
    'Skill_Retention_Score', 'Anxiety_Level_During_Exams', 'Prompt_Engineering_Skill',
]
ordem = [
    'Persona A: Mestre em Prompts',
    'Persona B: Dependente Hiperconectado',
    'Persona C: Tradicionalista Dedicado',
    'Persona D: Desengajado em Risco',
]

# (a) Tabela 2: N, proporção, médias brutas e ΔGPA médio por persona
tabela2 = df.groupby('persona')[cols_cluster + ['delta_gpa']].mean().loc[ordem]
tabela2.insert(0, 'N', df['persona'].value_counts().loc[ordem])
tabela2.insert(1, 'Proporção (%)', (tabela2['N'] / len(df) * 100).round(2))
print(tabela2.round(3).T)
tabela2.round(4).to_csv('tabela_2_centroides_por_persona.csv')

# (b) Radar com z-scores calculados
z = (df[cols_cluster] - df[cols_cluster].mean()) / df[cols_cluster].std()
z['persona'] = df['persona']
centroides_z = z.groupby('persona')[cols_cluster].mean().loc[ordem]
print(centroides_z.round(2))
centroides_z.round(4).to_csv('centroides_z_por_persona.csv')

rotulos = ['Uso de IA\n(Horas)', 'Estudo Tradicional\n(Horas)', 'Dependência\nPercebida',
           'Retenção de\nHabilidades', 'Ansiedade em\nExames', 'Habilidade em\nPrompts']
N = len(cols_cluster)
angulos = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
cores = ['#1f77b4', '#8c564b', '#2ca02c', '#d62728']

plt.figure(figsize=(8, 8), dpi=300)
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angulos[:-1], rotulos, size=10, fontweight='bold')
ax.set_rlabel_position(0)
lim = float(np.ceil(np.abs(centroides_z.values).max() * 2) / 2)
plt.ylim(-lim, lim)

for nome, cor in zip(ordem, cores):
    v = centroides_z.loc[nome].tolist()
    v += v[:1]
    ax.plot(angulos, v, linewidth=2, label=nome, color=cor)
    ax.fill(angulos, v, cor, alpha=0.15)

plt.title('Perfil dos Centróides Padronizados (Z-Score) por Persona\n',
          size=12, fontweight='bold', pad=25)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig('figura_14_perfil_personas_radar_calculado.png', dpi=300, bbox_inches='tight')
plt.show()
