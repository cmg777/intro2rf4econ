import React, { useState, useMemo } from 'react';
import ScatterPlot from './components/ScatterPlot';
import DecisionTreeVisualizer from './components/DecisionTreeVisualizer';
import { INITIAL_DATA } from './constants';

const App: React.FC = () => {
  // Model Parameters (Visualized for Tree #1 - The "Visible" Tree)
  const [incomeThreshold, setIncomeThreshold] = useState(20);
  const [lowIncomeEducThreshold, setLowIncomeEducThreshold] = useState(12);
  const [highIncomeEducThreshold, setHighIncomeEducThreshold] = useState(16);

  // Simulation / "Test Student" Parameters
  const [simIncome, setSimIncome] = useState<number>(25);
  const [simEduc, setSimEduc] = useState<number>(14);
  const [isSimulating, setIsSimulating] = useState(false);

  // Reset Handler
  const handleReset = () => {
    setIncomeThreshold(20);
    setLowIncomeEducThreshold(12);
    setHighIncomeEducThreshold(16);
    setSimIncome(25);
    setSimEduc(14);
    setIsSimulating(false);
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

        {/* NEW: Detailed User Guide */}
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            How to Use This App
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-blue-800">
            <div>
                <span className="font-bold block mb-1 text-blue-900">1. Be the Algorithm (Training)</span>
                <p className="opacity-90 leading-relaxed">
                  The sliders at the top control <strong>Tree #1</strong>. Your goal is to find rules that separate the green dots (Found Job) from red dots (No Job). Moving the "Income" slider shifts the dashed line on the plot; moving "Edu" sliders changes the vertical lines.
                </p>
            </div>
            <div>
                <span className="font-bold block mb-1 text-blue-900">2. Observe the Logic (Visualization)</span>
                <p className="opacity-90 leading-relaxed">
                  As you move the sliders, the <strong>Scatter Plot</strong> (left) and <strong>Decision Tree</strong> (right) update instantly. The scatter plot shows the <em>zones</em> of decision, while the tree shows the <em>rules</em> (e.g., "Is Income < 20k?").
                </p>
            </div>
            <div>
                <span className="font-bold block mb-1 text-blue-900">3. Run the Forest (Prediction)</span>
                <p className="opacity-90 leading-relaxed">
                  Scroll down to the <strong>Simulator</strong>. When you toggle it ON, you act as a new student. You'll see how <strong>Tree #1</strong> (your rules) votes alongside two "Ghost Trees" (hidden computer models) to reach a consensus.
                </p>
            </div>
            </div>
        </div>

        {/* Introduction Section with Analogy */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
          <div className="flex items-start gap-4">
             <div className="bg-emerald-100 p-3 rounded-full hidden sm:block">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
               </svg>
             </div>
             <div>
                <h2 className="text-xl font-bold text-slate-800 mb-3">Random Forest: The Wisdom of Crowds</h2>
                <div className="text-slate-600 text-sm leading-relaxed space-y-4">
                  <p>
                    Imagine you want to guess how many jellybeans are in a large jar. A single person's guess might be way off. But if you ask 100 people and take the average of their guesses, the result is usually surprisingly accurate.
                  </p>
                  <p>
                    <strong>Random Forest</strong> applies this concept to data. Instead of relying on one "expert" (a single Decision Tree) that might over-analyze the data (overfitting), we build a "committee" of hundreds of trees. Each tree sees a slightly different version of the data, so they make different mistakes. When they vote together, the mistakes cancel out, leaving a robust prediction.
                  </p>
                </div>
             </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 pt-6 border-t border-slate-100">
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="font-bold text-slate-800 block mb-1 text-sm">1. Bagging (Bootstrap)</span>
              <p className="text-xs text-slate-500">We create 100 different training datasets by sampling the original data randomly "with replacement". Some trees might see a specific data point 3 times, others not at all.</p>
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="font-bold text-slate-800 block mb-1 text-sm">2. Feature Randomness</span>
              <p className="text-xs text-slate-500">At each split, a tree is only allowed to look at a random subset of features (e.g., only "Income"). This forces trees to be diverse and independent.</p>
            </div>
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
              <span className="font-bold text-slate-800 block mb-1 text-sm">3. Majority Voting</span>
              <p className="text-xs text-slate-500">If 70 trees vote "Found Job" and 30 vote "No Job", the Forest predicts "Found Job". This "smoothing" effect reduces the variance of the model.</p>
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

        {/* Visual Connection Note */}
        <div className="text-center text-xs text-slate-400 mt-2 mb-8 italic">
           The <strong>Tree Logic</strong> (right) creates the <strong>Decision Boundaries</strong> (left). Moving the sliders changes the logic, which instantly redraws the map.
        </div>

        {/* Bottom Section: Educational Content & Simulator */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
           
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
                  This accuracy score applies <strong>only</strong> to the single tree you are controlling above. A full Random Forest would generally achieve higher accuracy by correcting the mistakes of this single tree.
                </p>
              </div>
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
              <h3 className="font-bold text-slate-800 text-lg mb-2">Why the "Ghost Trees"?</h3>
              <p className="text-sm text-slate-600 leading-relaxed mb-4">
                In the simulator, <strong>Tree #2</strong> and <strong>Tree #3</strong> are "Ghost Trees". They represent other trees in the forest that were trained on <em>slightly different data</em> (via Bagging).
              </p>
              <div className="bg-slate-50 p-3 rounded-lg text-xs text-slate-500 border border-slate-100 space-y-2">
                 <p>Because they saw different data, they learned slightly different rules (thresholds).</p>
                 <p><strong>The Payoff:</strong> Even if your visible tree makes a mistake on a tricky student, the other trees might get it right. The majority vote protects the model from individual errors.</p>
              </div>
           </div>
        </div>
      </main>
    </div>
  );
};

export default App;