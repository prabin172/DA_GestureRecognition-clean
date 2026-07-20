# SYSTEM DIRECTIVE: Manuscript and Code Refactoring for IEEE THMS

You are to execute a major structural and narrative overhaul of the controller simulation methodology and results for the current manuscript. This refactor targets the human-machine systems (THMS) narrative, abstracting the simulation away from specific kinematics and expanding the robustness checks across all domain gaps.

Please update the relevant Python evaluation scripts, and rewrite `method.tex` (Section 8) and `results.tex` (Section R5) according to the precise instructions below.

## 1. Global Terminology Scrub
Eliminate all semantic human-factors terminology that distracts from the core mathematical compounding of errors. 
*   **Remove:** "Pick-and-place mission", "Command vocabulary", "Grasp", "Release", "Cancel"[cite: 6].
*   **Replace with:** "12-step Sequential Control Task", "Action Primitives" (or "System Inputs"), "Safety-Critical States", "Routine States", and "Recoverable Errors"[cite: 6].
*   **Definition of States:** A "Safety-Critical State" is defined strictly as a state where a misclassification or false activation triggers immediate task failure[cite: 6]. A "Routine State" allows recoverable errors with a time/effort penalty[cite: 6]. Do not map specific physical movements (e.g., squat, bow) to these states[cite: 6].

## 2. Code Execution & Data Generation Protocol
The simulation must transition from a single-gap "illustrative" test to a multi-gap generalized proof. Update the scripts and generate the following data:
*   **Drop the Illustrative Run:** Remove the prototype run that manually mapped high-recall gestures to critical states[cite: 6, 7]. 
*   **Calculate ECE for All Gaps:** Calculate the Expected Calibration Error (ECE) for the Small Gaps (CZU/UTD Skeleton) and Huge Gaps (CZU IMU Orientation-only and Dual Raw)[cite: 6, 7].
*   **Run the Controller Across All Gaps:** Execute the simulation on the posteriors from all three gaps: Small (CZU/UTD), Moderate (Xsens), and Huge (CZU IMU)[cite: 6, 7].
*   **Enforce Robust Parameters Only:** Lock the execution to the Monte Carlo approach. All reported results must be the average of 120 randomized gesture-to-state mappings, evaluated through the critical-cost sweep and the iso-safety threshold[cite: 6, 7].

## 3. Rewriting Section 8 (Methods - Controller Simulation)
Rewrite the controller simulation methodology to frame it as a generalized mathematical evaluation of system reliability. 
*   **FSM Abstraction:** State that to translate recognition accuracy and ECE into human-machine systems metrics, the models' posterior streams are evaluated through an abstract Finite State Machine (FSM)[cite: 6]. Define the states mathematically as requested in Step 1.
*   **The Monte Carlo Protocol:** Present the 120-vocabulary randomization (previously "Lock 1") as the *primary evaluation protocol*, ensuring findings are fundamental to the objective, not an artifact of interface design[cite: 6].
*   **Iso-Safety Operating Limits:** Introduce the threshold ($\tau$) mechanism. Explain that because real-world systems operate on strict false-activation budgets, throughput is evaluated at a fixed iso-safety operating point (e.g., 1%) rather than a static accuracy threshold[cite: 6].

## 4. Rewriting Section R5 (Results - Controller)
Restructure the results narrative to tell a complete story about how calibration and accuracy degrade under shift, ordered by gap size:
*   **The Small Gap (Skeleton $\rightarrow$ Skeleton):** Show that high accuracy translates directly to high task success. Highlight that `supLP120` dominates simply because its raw recognition is superior[cite: 7].
*   **The Moderate Gap (Skeleton $\rightarrow$ Xsens IMU) - *The Core Finding*:** Emphasize that while raw accuracy compresses to a narrow spread[cite: 1, 7], the controller exposes massive reliability differences. Present the 120-vocabulary average[cite: 7]. Prove the "Calibration Link": use the iso-safety threshold data to demonstrate that `supLP120` requires the lowest safety threshold ($\tau=0.90$) due to its superior ECE, yielding the highest throughput and lowest task cost[cite: 7]. Show that `mae` compounds worst[cite: 7].
*   **The Huge Gap (Skeleton $\rightarrow$ CZU IMU):** Document the collapse. Show that the supervised prior (`supLP120`) triggers catastrophic mission failure across the randomized vocabularies because its rigid boundaries misalign with the target data[cite: 7]. Cross-reference this with its ECE at this gap to discuss the nature of its failure (confident vs. uncertain).

Execute these changes and output the revised `.tex` sections and a summary of the modified Python code.
