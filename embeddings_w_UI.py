import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

st.title("Record Linkage with Embeddings")

# File upload
uploaded_dirty = st.file_uploader("Upload first CSV (dirty list)", type="csv")
uploaded_clean = st.file_uploader("Upload second CSV (clean list)", type="csv")

if uploaded_dirty and uploaded_clean:
    df_dirty = pd.read_csv(uploaded_dirty)
    df_clean = pd.read_csv(uploaded_clean)

    # Select the column to match on
    dirty_col = st.selectbox("Select column from first CSV", df_dirty.columns)
    clean_col = st.selectbox("Select column from second CSV", df_clean.columns)

    # Set similarity threshold
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.01)

    if st.button("Run Linkage"):
        st.write("Generating embeddings and computing similarities...")

        # Prepare data
        dirty_names = df_dirty[dirty_col].astype(str).tolist()
        clean_names = df_clean[clean_col].astype(str).tolist()
        
        st.write("Loading model SentenceTransformer('all-MiniLM-L6-v2')..")
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode
        dirty_emb = model.encode(dirty_names, convert_to_tensor=False)
        clean_emb = model.encode(clean_names, convert_to_tensor=False)

        st.write("Computing cosine similarity...")
        #Similarity matrix
        time.sleep(1)  # Simulate some processing time
        sim_matrix = cosine_similarity(dirty_emb, clean_emb)

        # Find matches above threshold
        rows, cols = np.where(sim_matrix >= threshold)
        results = []
        for i, j in zip(rows, cols):
            results.append({
                f"{dirty_col} (dirty)": dirty_names[i],
                f"{clean_col} (clean)": clean_names[j],
                "Similarity": sim_matrix[i, j]
            })

        if results:

            results_df = pd.DataFrame(results).sort_values("Similarity", ascending=False)
            st.success(f"Found {len(results)} matches above threshold. {len(dirty_names)-len(results)} dirty names remain unmatched.")
            st.dataframe(results_df)

            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="record_linkage_results.csv",
                mime="text/csv")
        else:
            st.warning("No matches found above the threshold.")

    st.markdown("---")
        # Restart button logic
    if st.button("Restart"):
        st.experimental_rerun()
#     st.markdown("**Tip:** You can download the results as CSV after copying the table above into Excel or use Streamlit's download button with a few more lines of code.")



# # Compute cosine similarity between all pairs
# similarity_matrix = cosine_similarity(dirty_emb, clean_emb)

# # For each dirty name, find the best match in clean names
# best_match_indices = np.argmax(similarity_matrix, axis=1)
# best_match_scores = np.max(similarity_matrix, axis=1)

# # Set a similarity threshold (e.g., 0.7)
# threshold = 0.5
# count = 0
# print("Best matches above threshold:\n")
# for i, (idx, score) in enumerate(zip(best_match_indices, best_match_scores)):
#     if score >= threshold:
#         print(f"{dirty_names[i]}  <==>  {clean_names[idx]}  (similarity: {score:.2f})")
#         count += 1

# print(f"\nTotal matches above threshold: {count}")
# print(f"Total dirty names: {len(dirty_names)}")
# # Save results to a CSV file
# results = pd.DataFrame({
#     'dirty_name': dirty_names,
#     'best_match': [clean_names[idx] for idx in best_match_indices],
#     'similarity_score': best_match_scores
# })
# results = results[results['similarity_score'] >= threshold]
# results.to_csv('match_results.csv', index=False)
# print("Results saved to 'match_results.csv'.")
