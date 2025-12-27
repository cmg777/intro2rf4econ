import React, { useState, useMemo } from 'react';
import ScatterPlot from './components/ScatterPlot';
import DecisionTreeVisualizer from './components/DecisionTreeVisualizer';
import { INITIAL_DATA } from './constants';

const PYTHON_CODE = `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

# ==============================================================================
# PROFESSOR'S NOTE: THE RANDOM FOREST (The "Wisdom of Crowds")
# We are upgrading from a single "expert" (Decision Tree) to a "committee" (Forest).
#
# The Core Logic:
# 1. Bagging (Bootstrap Aggregating): We create 100 new datasets by sampling 
#    with replacement. Some data points appear multiple times, some not at all.
# 2. Independence: Each tree is trained on a different random subset of data 
#    and allowed to see different random subsets of features.
# 3. Voting: The final prediction is the average (or majority vote) of the 100 trees.
#    This reduces "Variance" (overfitting).
# ==============================================================================

# ==========================================
# STEP 1: DATA GENERATION (Same Sample)
# ==========================================
data = [
    # -- High Income (>= $20k), High Education (>= 16y) -> Job Found --
    {'education': 18, 'income': 80, 'found_job': 1},
    {'education': 17, 'income': 70, 'found_job': 1},
    {'education': 19, 'income': 40, 'found_job': 1},
    {'education': 16.5, 'income': 60, 'found_job': 1},
    
    # -- High Income (>= $20k), Low Education (< 16y) -> No Job --
    {'education': 5, 'income': 85, 'found_job': 0},
    {'education': 8, 'income': 75, 'found_job': 0},
    {'education': 4, 'income': 65, 'found_job': 0},
    {'education': 12, 'income': 80, 'found_job': 0},
    {'education': 10, 'income': 55, 'found_job': 0},
    {'education': 14, 'income': 70, 'found_job': 0},
    {'education': 2, 'income': 45, 'found_job': 0},
    {'education': 7, 'income': 30, 'found_job': 0},

    # -- Low Income (< $20k), High Education (>= 12y) -> Job Found --
    {'education': 14, 'income': 15, 'found_job': 1},
    {'education': 16, 'income': 10, 'found_job': 1},
    {'education': 18, 'income': 5, 'found_job': 1},
    {'education': 12, 'income': 8, 'found_job': 1},
    {'education': 15, 'income': 12, 'found_job': 1},
    {'education': 19, 'income': 2, 'found_job': 1},
    {'education': 13, 'income': 18, 'found_job': 1},

    # -- Low Income (< $20k), Low Education (< 12y) -> No Job --
    {'education': 4, 'income': 15, 'found_job': 0},
    {'education': 8, 'income': 10, 'found_job': 0},
    {'education': 2, 'income': 5, 'found_job': 0},
    {'education': 6, 'income': 12, 'found_job': 0},
    {'education': 1, 'income': 18, 'found_job': 0},
]

df = pd.DataFrame(data)
X = df[['income', 'education']] 
y = df['found_job']

# ==========================================
# STEP 2: IDENTIFICATION (Train-Test Split)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# STEP 3: INITIALIZING THE FOREST
# ==========================================
# n_estimators=100: We are hiring a committee of 100 voters.
# max_depth=3: We intentionally keep voters "weak" (shallow) to prevent one 
#              opinion from dominating.
# bootstrap=True: Each voter sees a different slice of the history.
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=3, 
    bootstrap=True,
    random_state=42
)

# ==========================================
# STEP 4: TRAINING (The Ensemble)
# ==========================================
# Scikit-Learn fits 100 trees in parallel.
rf_model.fit(X_train, y_train)

# ==========================================
# STEP 5: PREDICTION & EVALUATION
# ==========================================
y_pred = rf_model.predict(X_test)

print("--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Job', 'Found Job']))

# ==========================================
# STEP 6: FEATURE IMPORTANCE (The Economic Insight)
# ==========================================
# Professor's Note: We lose the single "Tree Diagram" because there are now 100.
# Instead, we ask: "Which variable did the committee discuss the most?"
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title("Feature Importance: Which variable drives the decision?")
plt.xlabel("Importance Score (0 to 1)")
plt.show()

# ==========================================
# STEP 7: VISUALIZING A SINGLE VOTER
# ==========================================
# We can peek inside the black box to see what ONE of the 100 trees looks like.
single_tree = rf_model.estimators_[0]

plt.figure(figsize=(12, 6))
plot_tree(single_tree, 
          feature_names=['Income', 'Education'], 
          class_names=['No Job', 'Found Job'], 
          filled=True, rounded=True)
plt.title("Tree #1 out of 100 (Just one voter's opinion)")
plt.show()

# ==========================================
# STEP 8: VISUALIZING THE CONSENSUS (Decision Boundary)
# ==========================================
# Compare this to the single tree plot. The boundaries should look 
# slightly more complex or "nuanced" because 100 trees are averaging their lines.
plt.figure(figsize=(10, 6))

x_min, x_max = 0, 100
y_min, y_max = 0, 20
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = rf_model.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlGn')

# Plot Testing Data (Stars)
test_df = pd.concat([X_test, y_test], axis=1)
plt.scatter(test_df[test_df['found_job']==1]['income'], test_df[test_df['found_job']==1]['education'], 
            c='green', marker='*', s=200, edgecolors='k', label='Test: Job')
plt.scatter(test_df[test_df['found_job']==0]['income'], test_df[test_df['found_job']==0]['education'], 
            c='red', marker='*', s=200, edgecolors='k', label='Test: No Job')

plt.ylabel('Education (Years)')
plt.xlabel('Prior Income (k$)')
plt.title('Random Forest Decision Boundary (The Committee Vote)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`;

const App: React.FC = () => {
  // Model Parameters (Visualized for Tree #1 - The "Visible" Tree)
  const [incomeThreshold, setIncomeThreshold] = useState(20);
  const [lowIncomeEducThreshold, setLowIncomeEducThreshold] = useState(12);
  const [highIncomeEducThreshold, setHighIncomeEducThreshold] = useState(16);

  // Simulation / "Test Student" Parameters
  const [simIncome, setSimIncome] = useState<number>(25);
  const [simEduc, setSimEduc] = useState<number>(14);
  const [isSimulating, setIsSimulating] = useState(false);

  // Copy State
  const [copied, setCopied] = useState(false);

  // Reset Handler
  const handleReset = () => {
    setIncomeThreshold(20);
    setLowIncomeEducThreshold(12);
    setHighIncomeEducThreshold(16);
    setSimIncome(25);
    setSimEduc(14);
    setIsSimulating(false);
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(PYTHON_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // 1. Calculate Single Tree Accuracy (The visible one)
  const stats = useMemo(() => {
    let correctCount = 0;
    INITIAL_DATA.forEach(point => {
      let predictedJob = false;
      if (point.income < incomeThreshold) {
        predictedJob = point.education >= lowIncomeEducThreshold;
      } else {
        predictedJob = point.education >= highIncomeEducThreshold;
      }
      if (predictedJob === point.foundJob) correctCount++;
    });
    const accuracy = Math.round((correctCount / INITIAL_DATA.length) * 100);
    return { accuracy, correctCount, total: INITIAL_DATA.length };
  }, [incomeThreshold, lowIncomeEducThreshold, highIncomeEducThreshold]);

  // 2. Simulate Random Forest Voting (3 Trees)
  // Tree 1: The user controlled tree.
  // Tree 2: "Ghost" tree with slightly lower thresholds (simulating different data bag)
  // Tree 3: "Ghost" tree with slightly higher thresholds
  const votingResult = useMemo(() => {
    if (!isSimulating) return null;

    const runTree = (inc: number, edu: number, tInc: number, tLow: number, tHigh: number) => {
      if (inc < tInc) return edu >= tLow;
      return edu >= tHigh;
    };

    // Tree 1 (Visible)
    const vote1 = runTree(simIncome, simEduc, incomeThreshold, lowIncomeEducThreshold, highIncomeEducThreshold);
    
    // Tree 2 (Ghost - Variation A)
    // Simulates a tree trained on data where the optimal split was found slightly lower
    const vote2 = runTree(simIncome, simEduc, incomeThreshold - 5, lowIncomeEducThreshold + 1, highIncomeEducThreshold - 1);

    // Tree 3 (Ghost - Variation B)
    // Simulates a tree trained on data where the optimal split was found slightly higher
    const vote3 = runTree(simIncome, simEduc, incomeThreshold + 5, lowIncomeEducThreshold - 1, highIncomeEducThreshold + 2);

    const yesVotes = [vote1, vote2, vote3].filter(v => v).length;
    const finalPrediction = yesVotes >= 2;

    return {
      tree1: vote1,
      tree2: vote2,
      tree3: vote3,
      yesVotes,
      finalPrediction
    };
  }, [simIncome, simEduc, isSimulating, incomeThreshold, lowIncomeEducThreshold, highIncomeEducThreshold]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-12">
      {/* Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-600 rounded flex items-center justify-center">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <span className="font-bold text-slate-800 text-lg tracking-tight">Econometrics<span className="text-emerald-600">Lab</span></span>
          </div>
          <div className="text-xs text-slate-500 hidden sm:block">
            Introduction to Random Forest
          </div>
        </div>
      </nav>

      {/* Main Controls Toolbar */}
      <div className="bg-white border-b border-slate-200 py-4 px-4 shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row gap-6 items-center justify-center">
          <div className="flex items-center gap-4 w-full md:w-auto">
             <span className="text-xs font-bold text-slate-400 uppercase tracking-wide w-24">Tree #1 Split</span>
             <div className="flex flex-col w-full md:w-48">
               <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                 <span>Income</span>
                 <span className="text-blue-600">${incomeThreshold}k</span>
               </div>
               <input 
                  type="range" min="10" max="60" value={incomeThreshold} 
                  onChange={(e) => setIncomeThreshold(Number(e.target.value))}
                  className="h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
             </div>
          </div>

          <div className="w-px h-8 bg-slate-200 hidden md:block"></div>

          <div className="flex items-center gap-4 w-full md:w-auto">
             <span className="text-xs font-bold text-slate-400 uppercase tracking-wide w-24">Tree #1 Leaves</span>
             
             <div className="flex flex-col w-full md:w-48">
               <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                 <span>Edu (Low Inc)</span>
                 <span className="text-blue-600">{lowIncomeEducThreshold}y</span>
               </div>
               <input 
                  type="range" min="8" max="16" value={lowIncomeEducThreshold} 
                  onChange={(e) => setLowIncomeEducThreshold(Number(e.target.value))}
                  className="h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
             </div>

             <div className="flex flex-col w-full md:w-48 ml-4">
               <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                 <span>Edu (High Inc)</span>
                 <span className="text-blue-600">{highIncomeEducThreshold}y</span>
               </div>
               <input 
                  type="range" min="12" max="18" value={highIncomeEducThreshold} 
                  onChange={(e) => setHighIncomeEducThreshold(Number(e.target.value))}
                  className="h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
             </div>
          </div>

          <div className="w-px h-8 bg-slate-200 hidden md:block"></div>

          <button 
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-slate-500 bg-slate-50 hover:bg-slate-100 hover:text-emerald-600 rounded-lg border border-slate-200 transition-all shadow-sm active:scale-95"
            title="Reset to default values"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            RESET
          </button>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-8">

        {/* NEW: Guide / Start Here Section */}
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            How to Use This App
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-blue-800">
            <div>
                <span className="font-bold block mb-1">1. Tweak the Base Learner</span>
                <p className="opacity-90">Use the sliders in the top toolbar to define the rules for <strong>Tree #1</strong>. Watch the <span className="font-semibold">Scatter Plot</span> (left) and <span className="font-semibold">Decision Tree</span> (right) update in real-time.</p>
            </div>
            <div>
                <span className="font-bold block mb-1">2. Test the Forest</span>
                <p className="opacity-90">Scroll down to the <strong>Forest Voting Simulator</strong> and toggle it <strong>ON</strong>. This creates a "Mini-Forest" with two hidden "ghost trees" to demonstrate ensemble voting.</p>
            </div>
            <div>
                <span className="font-bold block mb-1">3. Learn the Concept</span>
                <p className="opacity-90">Discover how <strong>Random Forests</strong> use bagging and majority voting to outperform single decision trees, then grab the Python code to try it yourself.</p>
            </div>
            </div>
        </div>

        {/* Introduction Section */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-bold text-slate-800 mb-3">Random Forest: The Power of the Crowd</h2>
          <div className="text-slate-600 text-sm leading-relaxed space-y-4">
            <p>
              A single decision tree is powerful but prone to errors (high variance). If you change the training data slightly, the tree structure might change completely. 
              The <strong>Random Forest</strong> method solves this by training hundreds of trees on random subsets of the data and averaging their predictions.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <span className="font-bold text-slate-800 block mb-1">1. Bagging</span>
                <p className="text-xs">Short for "Bootstrap Aggregating". We create many "random" datasets by sampling from the original data with replacement.</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <span className="font-bold text-slate-800 block mb-1">2. Decorrelation</span>
                <p className="text-xs">Each tree is forced to be different, ensuring that the forest has a diverse set of opinions.</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <span className="font-bold text-slate-800 block mb-1">3. Majority Voting</span>
                <p className="text-xs">If 70 trees say "Job Found" and 30 say "No Job", the forest predicts "Job Found". The interactive tool below demonstrates this.</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-auto">
           {/* Left: Data View */}
           <div className="h-[500px]">
             <ScatterPlot 
              data={INITIAL_DATA} 
              incomeThreshold={incomeThreshold}
              lowIncomeEducThreshold={lowIncomeEducThreshold}
              highIncomeEducThreshold={highIncomeEducThreshold}
              simulationPoint={isSimulating ? { income: simIncome, education: simEduc } : null}
             />
           </div>

           {/* Right: Logic View */}
           <div className="h-[500px]">
             <DecisionTreeVisualizer 
              incomeThreshold={incomeThreshold}
              lowIncomeEducThreshold={lowIncomeEducThreshold}
              highIncomeEducThreshold={highIncomeEducThreshold}
              simulationPoint={isSimulating ? { income: simIncome, education: simEduc } : null}
             />
           </div>
        </div>

        {/* Bottom Section: Educational Content & Simulator */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-12 gap-6">
           
           {/* Accuracy Card */}
           <div className="md:col-span-4 bg-white p-5 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-slate-800 text-lg mb-2">Base Learner Accuracy</h3>
                <div className="flex items-center gap-3 mb-2">
                    <div className="flex-1 h-3 bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-green-500 rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${stats.accuracy}%` }}
                      ></div>
                    </div>
                    <span className="text-lg font-bold text-slate-700">{stats.accuracy}%</span>
                </div>
                <p className="text-xs text-slate-500">
                  This accuracy reflects only the single tree shown above. A full forest would typically achieve higher accuracy by reducing errors.
                </p>
              </div>
              <p className="text-sm text-slate-600 mt-4 leading-relaxed">
                Try moving the sliders to "train" this single tree. In a real Random Forest, the computer does this automatically 100 times!
              </p>
           </div>

           {/* Simulator Card (VOTING) */}
           <div className="md:col-span-4 bg-emerald-50 border border-emerald-100 p-5 rounded-xl shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-slate-800 text-lg">Forest Voting Simulator</h3>
                <label className="flex items-center cursor-pointer">
                  <div className="relative">
                    <input type="checkbox" className="sr-only" checked={isSimulating} onChange={() => setIsSimulating(!isSimulating)} />
                    <div className={`block w-10 h-6 rounded-full transition-colors ${isSimulating ? 'bg-emerald-600' : 'bg-slate-300'}`}></div>
                    <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${isSimulating ? 'transform translate-x-4' : ''}`}></div>
                  </div>
                  <span className="ml-2 text-xs font-semibold text-slate-600">{isSimulating ? 'ON' : 'OFF'}</span>
                </label>
              </div>
              
              <div className={`space-y-4 transition-opacity ${isSimulating ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                 {/* Inputs */}
                 <div className="grid grid-cols-2 gap-4">
                   <div>
                     <label className="block text-[10px] font-bold text-slate-500 mb-1 uppercase">Income</label>
                     <input type="range" min="0" max="100" value={simIncome} onChange={(e) => setSimIncome(Number(e.target.value))} className="w-full h-1.5 bg-emerald-200 rounded-lg appearance-none cursor-pointer accent-emerald-600" />
                     <div className="text-right text-xs font-bold text-emerald-700">${simIncome}k</div>
                   </div>
                   <div>
                     <label className="block text-[10px] font-bold text-slate-500 mb-1 uppercase">Education</label>
                     <input type="range" min="0" max="20" step="0.5" value={simEduc} onChange={(e) => setSimEduc(Number(e.target.value))} className="w-full h-1.5 bg-emerald-200 rounded-lg appearance-none cursor-pointer accent-emerald-600" />
                     <div className="text-right text-xs font-bold text-emerald-700">{simEduc} yrs</div>
                   </div>
                 </div>

                 {/* Voting Results */}
                 {votingResult && (
                   <div className="bg-white p-3 rounded-lg border border-emerald-100">
                     <div className="text-xs font-bold text-slate-400 mb-2 uppercase tracking-wide">Committee Votes (3 Trees)</div>
                     
                     <div className="space-y-2 mb-3">
                       <div className="flex justify-between text-xs items-center">
                         <span className="text-slate-600 font-medium">Tree #1 (Visible):</span>
                         <span className={`font-bold ${votingResult.tree1 ? 'text-green-600' : 'text-red-600'}`}>{votingResult.tree1 ? 'Job Found' : 'No Job'}</span>
                       </div>
                       <div className="flex justify-between text-xs items-center opacity-70">
                         <span className="text-slate-500 font-medium">Tree #2 (Ghost):</span>
                         <span className={`font-bold ${votingResult.tree2 ? 'text-green-600' : 'text-red-600'}`}>{votingResult.tree2 ? 'Job Found' : 'No Job'}</span>
                       </div>
                       <div className="flex justify-between text-xs items-center opacity-70">
                         <span className="text-slate-500 font-medium">Tree #3 (Ghost):</span>
                         <span className={`font-bold ${votingResult.tree3 ? 'text-green-600' : 'text-red-600'}`}>{votingResult.tree3 ? 'Job Found' : 'No Job'}</span>
                       </div>
                     </div>

                     <div className="pt-2 border-t border-slate-100 flex justify-between items-center">
                        <span className="text-sm font-bold text-slate-800">Final Consensus:</span>
                        <span className={`px-2 py-1 rounded text-xs font-bold ${votingResult.finalPrediction ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                          {votingResult.finalPrediction ? 'JOB FOUND' : 'NO JOB'}
                        </span>
                     </div>
                   </div>
                 )}
              </div>
           </div>

           {/* Education Card */}
           <div className="md:col-span-4 bg-white p-5 rounded-xl shadow-sm border border-slate-200">
              <h3 className="font-bold text-slate-800 text-lg mb-2">Why Random Forest?</h3>
              <p className="text-sm text-slate-600 leading-relaxed mb-4">
                Did you see the "Ghost Trees" in the simulator? They vote differently because they would have been trained on slightly different data.
              </p>
              <div className="bg-slate-50 p-3 rounded-lg text-xs text-slate-500 border border-slate-100">
                 By taking the majority vote, the Random Forest cancels out the errors of individual trees, leading to a much more reliable prediction than any single tree could provide.
              </div>
           </div>
        </div>

        {/* Google Colab Section */}
        <div className="mt-12 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="bg-slate-800 p-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-xl font-bold text-white mb-1">Run Random Forest in Python</h2>
                    <p className="text-slate-400 text-sm">See how Scikit-Learn implements Bagging and Voting in just a few lines.</p>
                </div>
                
                <div className="flex gap-3">
                  <a 
                    href="https://colab.research.google.com/notebooks/empty.ipynb"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 border border-slate-600"
                  >
                     <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                     </svg>
                     Open Colab
                  </a>
                  
                  <button 
                    onClick={handleCopyCode}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2"
                  >
                    {copied ? (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        Copied!
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        Copy Code
                      </>
                    )}
                  </button>
                </div>
            </div>
            <div className="p-0 bg-[#1e1e1e] overflow-x-auto relative">
                <div className="absolute top-0 right-0 p-2 text-xs text-slate-500 font-mono">Python</div>
                <pre className="p-6 text-sm font-mono text-slate-300 leading-relaxed overflow-x-auto">
                    {PYTHON_CODE}
                </pre>
            </div>
        </div>

      </main>
    </div>
  );
};

export default App;