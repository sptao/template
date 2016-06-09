#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <stack>
#include <set>
#include <unordered_set>
#include <queue>
//#include "timer.h"

#pragma warning (disable: 4996)

using namespace std;

class Solution1 {
	class Point {
	public:
		int x;
		int y;
	};
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		vector<Point> vecPnt;
		for (int i = 0; i < nums.size(); i++) {
			Point p;
			p.x = nums[i];
			p.y = i + 1;
			vecPnt.push_back(p);
		}
		std::sort(vecPnt.begin(), vecPnt.end(), [](const Point &p1, const Point &p2){return p1.x < p2.x; });

		vector<int> ans(2, 0);
		for (int i = 0; i < vecPnt.size(); i++) {
			int deta = target - vecPnt[i].x;
			const Point *p = NULL;
			p = searchDeta(vecPnt, i + 1, vecPnt.size() - 1, deta);
			if (p != NULL) {
				if (vecPnt[i].y < p->y) {
					ans[0] = vecPnt[i].y;
					ans[1] = p->y;
				}
				else {
					ans[0] = p->y;
					ans[1] = vecPnt[i].y;										
				}
				break;
			}
		}

		return ans;
	}

	const Point *searchDeta(const vector<Point> &vp, int low, int high, int val) {
		if (low <= high) {
			int mid = (low + high) / 2;
			if (vp[mid].x == val) {
				return &vp[mid];
			}
			else if (vp[mid].x > val) {
				return searchDeta(vp, low, mid - 1, val);
			}
			else {
				return searchDeta(vp, mid + 1, high, val);
			}
		}

		return NULL;
	}
};

class Solution1_1 {
	class Point {
	public:
		int x;
		int y;
	};

public:
	vector<int> twoSum(vector<int>& nums, int target) {
		vector<Point> vp;
		for (int i = 0; i < nums.size(); i++) {
			Point p;
			p.x = i + 1;
			p.y = nums[i];
			vp.push_back(p);
		}
		sort(vp.begin(), vp.end(), [](const Point &a, const Point &b){return a.y < b.y; });
		int sum = target + 1;
		int i = 0;
		int j = vp.size() - 1;
		while (i < j && sum != target) {
			sum = vp[i].y + vp[j].y;
			if (sum > target) {
				j--;
			}
			else if (sum < target) {
				i++;
			}
		}
		vector<int> ans;
		if (vp[i].x < vp[j].x) {
			ans.push_back(vp[i].x);
			ans.push_back(vp[j].x);
		}
		else {
			ans.push_back(vp[j].x);
			ans.push_back(vp[i].x);
		}

		return ans;
	}
};

class Solution1_2 {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		unordered_multimap<int, int> umap;
		for (int i = 0; i < nums.size(); i++) {
			umap.insert(make_pair(nums[i], i));
		}
		
		vector<int> ans;
		for (int i = 0; i < nums.size() - 1; i++) {
			int d = target - nums[i];
			auto range = umap.equal_range(d);
			if (range.first != range.second) {
				int idx1 = i + 1;
				if (d == nums[i]) {
					idx1 = range.first->second + 1;
					if ((++range.first) == range.second) {
						continue;
					}
				}
				int idx2 = range.first->second + 1;				
				ans.push_back(min(idx1, idx2));
				ans.push_back(max(idx1, idx2));
				break;
			}			
		}

		return ans;
	}
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution2 {
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		ListNode *p = new ListNode(0);
		ListNode *r = p;
		int carry = 0;
		while (l1 != NULL && l2 != NULL) {
			int tmp = l1->val + l2->val + carry;
			carry = tmp / 10;
			tmp %= 10;
			r->next = new ListNode(tmp);
			r = r->next;
			l1 = l1->next;
			l2 = l2->next;
		}
		while (l1 != NULL) {
			int tmp = l1->val + carry;
			carry = tmp / 10;
			tmp %= 10;
			r->next = new ListNode(tmp);
			r = r->next;
			l1 = l1->next;
		}
		while (l2 != NULL) {
			int tmp = l2->val + carry;
			carry = tmp / 10;
			tmp %= 10;
			r->next = new ListNode(tmp);
			r = r->next;
			l2 = l2->next;
		}
		if (carry != 0) {
			r->next = new ListNode(carry);
		}

		r = p->next;
		delete p;
		return r;
	}
};

class Solution3 {
public:
	int lengthOfLongestSubstring(string s) {
		char mark[256] = { 0 };
		int cnt, m;
		int b = 1;
		mark[s[0]] = 1;
		m = 0;
		for (int i = 0; i < s.size(); i++) {			
			if (b < s.size()) {
				char c = s[b];
				while (mark[c] != 1) {
					mark[c] = 1;
					b++;
					if (b < s.size()) {
						c = s[b];
					}
				}
				cnt = b - i;
				mark[s[i]] = 0;
			}
			else {
				cnt = b - i;
			}

			if (cnt > m) {
				m = cnt;
			}
		}

		return m;
	}
};

class Solution15 {
public:
	vector<vector<int>> threeSum(vector<int>& nums1) {
		vector<vector<int>> ans;
		if (nums1.size() == 0) {
			return ans;
		}
		sort(nums1.begin(), nums1.end(), [](int a, int b){return a < b; });
		vector<int> nums;
		vector<int> m;
		int i = 0;
		while (i < nums1.size()) {
			int cnt = 1;
			while (i + cnt < nums1.size() && nums1[i + cnt] == nums1[i]) {
				cnt++;
			}
			nums.push_back(nums1[i]);
			m.push_back(cnt);
			i += cnt;
		}
		i = 0;
		for (; i < nums.size() && nums[i] < 0; i++) {
			if (m[i] > 1) {
				if (binary_search(nums.begin() + i + 1, nums.end(), -2 * nums[i])) {
					vector<int> vec;
					vec.push_back(nums[i]);
					vec.push_back(nums[i]);
					vec.push_back(-2 * nums[i]);
					ans.push_back(vec);
				}
			}
		}		//two same negative
		if (i < nums.size() && nums[i] == 0 && m[i] >= 3) {
			vector<int> vec;
			for (int i = 0; i < 3; i++) {
				vec.push_back(0);
			}
			ans.push_back(vec);
		}		//thres zeros
		for (i = nums.size() - 1; i >= 0 && nums[i] > 0; i--) {
			if (m[i] > 1) {
				if (binary_search(nums.begin(), nums.begin() + i, -2 * nums[i])) {
					vector<int> vec;
					vec.push_back(-2 * nums[i]);
					vec.push_back(nums[i]);
					vec.push_back(nums[i]);					
					ans.push_back(vec);
				}
			}
		}		//two same positive

		
		int a = 0, b = 0;
		while (b < nums.size() && nums[b] < 0) {
			b++;
		}
		while (a < nums.size() && nums[a] < 0 && b < nums.size()) {
			for (int i = b; i < nums.size(); i++) {
				int d = 0 - (nums[a] + nums[i]);
				if (binary_search(nums.begin() + a + 1, nums.begin() + i, d)) {
					vector<int> v;
					v.push_back(nums[a]);
					v.push_back(d);
					v.push_back(nums[i]);
					ans.push_back(v);
				}
			}
			a++;
		}		//three different

		return ans;
	}
};


class Solution15_1 {
public:
	vector<vector<int>> threeSum(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		vector<vector<int>> ans;
		for (int i = 0; i < nums.size() && nums[i] <= 0; i++) {
			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}
			int t = 0 - nums[i];
			int j = i + 1;
			int k = nums.size() - 1;
			while (j < k) {
				int sum = nums[j] + nums[k];
				if (sum < t) {
					j++;
				}
				else if (sum > t) {
					k--;
				}
				else {
					vector<int> one;
					one.push_back(nums[i]);
					one.push_back(nums[j]);
					one.push_back(nums[k]);
					ans.push_back(one);
					j++;
					while (j < nums.size() && nums[j] == nums[j - 1]) {
						j++;
					}
					k--;
					while (k > j && nums[k] == nums[k + 1]) {
						k--;
					}
				}
			}
		}

		return ans;
	}
};

class Solution16 {
public:
	int threeSumClosest(vector<int>& nums, int target) {
		sort(nums.begin(), nums.end());
		int md = ~(1 << 31);
		int ans;
		for (int i = 0; i < nums.size() - 2; i++) {
			int j = i + 1;
			int k = nums.size() - 1;
			int sum;
			while (j < k) {
				sum = nums[i] + nums[j] + nums[k];
				if (sum < target) {
					j++;
				}
				else if (sum > target) {
					k--;
				}
				else {
					return target;
				}
				int d = abs(target - sum);
				if (d < md) {
					md = d;
					ans = sum;
				}
			}

		}

		return ans;
	}
};

class Solution17 {
public:
	vector<string> letterCombinations(string digits) {
		vector<string> ans;
		if (digits.size() == 0) {
			return ans;
		}
		vector<string> strDigit;
		for (int i = 0; i < digits.size(); i++) {
			strDigit.push_back(digit2str(digits[i]));
		}
		vector<int> stack;
		for (int i = 0; i < strDigit.size(); i++) {
			stack.push_back(-1);
		}
		string one;
		int i = 0;
		while (i >= 0) {
			stack[i]++;
			if (stack[i] < strDigit[i].size()) {
				one.push_back(strDigit[i][stack[i]]);
				i++;
				if (i == strDigit.size()) {
					ans.push_back(one);
					one.pop_back();
					i--;
				}
			}
			else {
				if (one.size() != 0) {
					one.pop_back();
				}				
				stack[i] = -1;
				i--;
			}
		}

		return ans;
	}

private:
	string digit2str(char c) {
		switch (c) {
		case '2':
			return "abc";
		case '3':
			return "def";
		case '4':
			return "ghi";
		case '5':
			return "jkl";
		case '6':
			return "mno";
		case '7':
			return "pqrs";
		case '8':
			return "tuv";
		case '9':
			return "wxyz";
		default:
			return "";
		}
	}
};

class Solution19 {
public:
	ListNode* removeNthFromEnd(ListNode* head, int n) {
		ListNode *lnode = new ListNode(0);
		lnode->next = head;
		ListNode *p1, *p2;
		p1 = lnode;
		p2 = lnode;
		for (int i = 0; i < n; i++) {
			p2 = p2->next;
		}
		while (p2->next != NULL) {
			p1 = p1->next;
			p2 = p2->next;
		}
		p1->next = p1->next->next;
		return lnode->next;
	}
};

class Solution83 {
public:
	ListNode* deleteDuplicates(ListNode* head) {
		ListNode *p = head;
		while (p != NULL && p->next != NULL) {
			if (p->next->val == p->val) {
				p->next = p->next->next;
			}
			else {
				p = p->next;
			}			
		}

		return head;
	}
};

class Solution88 {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		vector<int> nums3;
		int i = 0, j = 0;
		while (i < m && j < n) {
			if (nums1[i] < nums2[j]) {
				nums3.push_back(nums1[i++]);
			}
			else {
				nums3.push_back(nums2[j++]);
			}
		}
		while (i < m) {
			nums3.push_back(nums1[i++]);
		}
		while (j < n) {
			nums3.push_back(nums2[j++]);
		}
		for (int i = 0; i < nums3.size(); i++) {
			nums1[i] = nums3[i];
		}
	}
};

class Solution278 {
public:
	int firstBadVersion(int n) {
		return binaryBadSearch(1, n);
	}

	bool isBadVersion(int version)
	{
		if (version >= 1702766719) {
			return true;
		}
		return false;
	}

private:
	int binaryBadSearch(int low, int high) {
		if (low < high) {
			long long m = ((long long)low + high) / 2;
			if (isBadVersion(m)) {
				return binaryBadSearch(low, m);
			}
			else {
				return binaryBadSearch(m + 1, high);
			}
		}
		else {
			return low;
		}
	}
};

class Solution242 {
public:
	bool isAnagram(string s, string t) {
		if (s.size() != t.size()) {
			return false;
		}
		char m1[256], m2[256];
		for (int i = 0; i < 256; i++) {
			m1[i] = 0;
			m2[i] = 0;
		}
		for (int i = 0; i < s.size(); i++) {
			m1[s[i]]++;
			m2[t[i]]++;
		}
		for (int i = 0; i < 256; i++) {
			if (m1[i] != m2[i]) {
				return false;
			}
		}

		return true;
	}
};

class Solution237 {
public:
	void deleteNode(ListNode* node) {
		if (node != NULL && node->next != NULL) {
			node->val = node->next->val;
			node->next = node->next->next;
		}
	}
};

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution226 {
public:
	TreeNode* invertTree(TreeNode* root) {
		if (root != NULL) {
			invertTree(root->left);
			invertTree(root->right);
			TreeNode *tmp = root->left;
			root->left = root->right;
			root->right = tmp;
		}
		
		return root;
	}
};

class Queue {
public:
	// Push element x to the back of queue.
	void push(int x) {
		if (flag != 0) {
			inv();
		}
		
		ss.push(x);
	}

	// Removes the element from in front of queue.
	void pop(void) {
		if (flag == 0) {
			inv();
		}
		
		ss.pop();
		
	}

	// Get the front element.
	int peek(void) {
		if (flag == 0) {
			inv();
		}
		
		return ss.top();
	}

	// Return whether the queue is empty.
	bool empty(void) {
		return ss.empty();
	}

	Queue() {
		while (!ss.empty()) {
			ss.pop();
		}
		flag = 0;
	}

private:
	stack<int> ss;
	int flag;

	void inv() {
		stack<int> st;
		while (!ss.empty()) {
			st.push(ss.top());
			ss.pop();
		}
		ss = st;
		flag = ~flag;
	}
};

class Solution231 {
public:
	bool isPowerOfTwo(int n) {
		if (n < 0) {
			return false;
		}
		int cnt = 0;
		while (n != 0) {
			if (n & 1) {
				cnt++;				
			}
			n >>= 1;
		}

		return (cnt == 1);
	}
};

typedef unsigned int uint32_t;

class Solution190 {
public:
	uint32_t reverseBits(uint32_t n) {
		uint32_t ans = 0;
		ans = ans | (n & 1);
		for (int i = 0; i < 31; i++) {
			ans <<= 1;			
			n >>= 1;
			ans = ans | (n & 1);			
		}

		return ans;
	}
};

class Solution160 {
public:
	ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
		int cnt1 = 0, cnt2 = 0;
		ListNode *h1 = new ListNode(0);
		h1->next = headA;
		ListNode *p1 = h1;
		while (p1->next != NULL) {
			p1 = p1->next;
			cnt1++;
		}
		ListNode *h2 = new ListNode(0);
		h2->next = headB;
		ListNode *p2 = h2;
		while (p2->next != NULL) {
			p2 = p2->next;
			cnt2++;
		}
		ListNode *ans = NULL;
		if (p1 == p2) {
			p1 = h1;
			while (cnt1 > cnt2) {
				p1 = p1->next;
				cnt1--;
			}
			p2 = h2;
			while (cnt1 < cnt2) {
				p2 = p2->next;
				cnt2--;
			}
			while (p1 != p2) {
				p1 = p1->next;
				p2 = p2->next;
			}
			ans = p1;
		}

		delete h1;
		delete h2;
		return ans;
	}
};

class Solution111 {
public:
	int minDepth(TreeNode* root) {
		vector<TreeNode *> nodes;
		nodes.push_back(root);		
		if (root == NULL) {
			return 0;
		}
		int n = 1;
		while (!nodes.empty()) {
			vector<TreeNode *> nodes2;
			for (int i = 0; i < nodes.size(); i++) {
				int isLeaf = true;
				if (nodes[i]->left != NULL) {
					nodes2.push_back(nodes[i]->left);
					isLeaf = false;
				}
				if (nodes[i]->right != NULL) {
					nodes2.push_back(nodes[i]->right);
					isLeaf = false;
				}
				if (isLeaf) {
					return n;
				}
			}

			nodes = nodes2;
			n++;
		}

		return -1;
	}
};

//Attention!!!
class Solution172 {
public:
	int trailingZeroes(int n) {
		if (n < 0) {
			return -1;
		}
		int cnt = 0;
		for (long long i = 5; n / i >= 1; i *= 5) {
			cnt += n / i;
		}

		return cnt;
	}
};

class Solution112 {
public:
	bool hasPathSum(TreeNode* root, int sum) {		
		if (root == NULL) {
			return false;
		}
		if (root->left == NULL && root->right == NULL) {
			if (sum == root->val) {
				return true;
			}
			else {
				return false;
			}
		}
		return (hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val));
	}
};


class Solution217 {
public:
	bool containsDuplicate(vector<int>& nums) {
		set<int> s;
		for (int i = 0; i < nums.size(); i++) {			
			if (s.find(nums[i]) != s.end()) {
				return true;
			}
			else {
				s.insert(nums[i]);
			}
		}

		return false;
	}
};

class Solution198 {
public:
	int rob(vector<int>& nums) {
		if (nums.size() == 0) {
			return 0;
		}
		if (nums.size() == 1) {
			return nums[0];
		}
		int an2 = nums[0];
		int an1 = max(nums[0], nums[1]);
		for (int i = 2; i < nums.size(); i++) {
			int an = max((nums[i] + an2), an1);
			an2 = an1;
			an1 = an;
		}

		return an1;
	}
};

class Solution238 {
public:
	vector<int> productExceptSelf(vector<int>& nums) {
		vector<int> ans(nums.size());
		int t = 1;
		for (int i = 0; i < nums.size(); i++) {
			ans[i] = t;
			t *= nums[i];
		}
		t = 1;
		for (int i = nums.size() - 1; i >= 0; i--) {
			ans[i] = ans[i] * t;
			t *= nums[i];
		}

		return ans;
	}
};

class Solution96 {
public:
	int numTrees(int n) {
		int *a = new int[n + 1];
		a[1] = 1;
		for (int i = 2; i <= n; i++) {
			a[i] = 2 * a[i - 1];
			for (int j = 1; j <= i - 2; j++) {
				a[i] += a[j] * a[i - 1 - j];
			}
		}

		int ans = a[n];
		delete[] a;
		return ans;
	}
};

class Solution35 {
public:
	int searchInsert(vector<int>& nums, int target) {
		return bSearch(nums, target, 0, nums.size() - 1);
	}

private:
	int bSearch(vector<int> &nums, int target, int low, int high) {
		if (low <= high) {
			int m = (low + high) / 2;
			if (nums[m] == target) {
				return m;
			}
			else if (nums[m] > target) {
				return bSearch(nums, target, low, m - 1);
			}
			else {
				return bSearch(nums, target, m + 1, high);
			}
		}
		return low;
	}
};


//Attention!!!
class Solution53 {
public:
	int maxSubArray(vector<int>& nums) {
		long long gm = -(~(1 << (8 * sizeof(int)-1))) - 1;
		long long lm = gm;
		for (int i = 0; i < nums.size(); i++) {
			lm = max(lm + nums[i], (long long)nums[i]);
			if (lm > gm) {
				gm = lm;
			}
		}

		return gm;
	}
};

class Solution108 {
public:
	TreeNode* sortedArrayToBST(vector<int>& nums) {
		return a2t(nums, 0, nums.size() - 1);
	}

private:
	TreeNode *a2t(vector<int>& nums, int low, int high) {
		TreeNode *r = NULL;
		if (low <= high) {
			int m = (low + high) / 2;
			r = new TreeNode(nums[m]);
			r->left = a2t(nums, low, m - 1);
			r->right = a2t(nums, m + 1, high);
		}

		return r;
	}
};

class Solution153 {
public:
	int findMin(vector<int>& nums) {
		return fm(nums, 0, nums.size() - 1);
	}

private:
	int fm(vector<int>& nums, int low, int high) {
		if (low < high && nums[low] > nums[high]) {
			int m = (low + high) / 2;
			if (nums[low] < nums[m]) {
				return fm(nums, m, high);
			}
			else if (nums[low] > nums[m]) {
				return fm(nums, low, m);
			}
			else {
				return nums[high];
			}
		}
		
		return nums[low];
	}
};

class Solution62 {
public:
	int uniquePaths(int m, int n) {
		if (m == 0 || n == 0) {
			return 0;
		}
		int *s = new int[n];
		for (int i = 0; i < n; i++) {
			s[i] = 1;
		}
		for (int i = 0; i < m - 1; i++) {
			for (int j = 1; j < n; j++) {
				s[j] += s[j - 1];
			}
		}

		int ans = s[n - 1];
		delete[] s;
		return ans;
	}
};

class Solution268 {
public:
	int missingNumber(vector<int>& nums) {
		int sum = (0 + nums.size()) * (nums.size() + 1) / 2;
		for (int i = 0; i < nums.size(); i++) {
			sum -= nums[i];
		}
		return sum;
	}
};

class Solution200 {
public:
	int numIslands(vector<vector<char>>& grid) {
		if (grid.size() == 0 || grid[0].size() == 0) {
			return 0;
		}

		rows = grid.size();
		cols = grid[0].size();
		int *num = new int[rows * cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				num[i * cols + j] = -1;
			}
		}
		int cnt = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (grid[i][j] == '1' && num[i * cols + j] == -1) {					
					mark(grid, num, i, j, cnt);
					cnt++;
				}
			}
		}

		delete[] num;
		return cnt;
	}

private:
	int rows = 0, cols = 0;
	void mark(vector<vector<char>>& grid, int *num, int i, int j, int cnt) {
		if (i < 0 || i >= rows || j < 0 || j >= cols || grid[i][j] == '0' || num[i * cols + j] != -1) {
			return;
		}
		num[i * cols + j] = cnt;
		mark(grid, num, i - 1, j, cnt);
		mark(grid, num, i + 1, j, cnt);
		mark(grid, num, i, j - 1, cnt);
		mark(grid, num, i, j + 1, cnt);
	}
};

class Solution222 {
public:
	int countNodes(TreeNode* root) {
		if (root == NULL) {
			return 0;
		}

		TreeNode *l = root->left, *r = root->right;
		int hl = 0, hr = 0;
		while (l != NULL) {
			l = l->left;
			hl++;
		}
		while (r != NULL) {
			r = r->right;
			hr++;
		}
		if (hl == hr) {
			return ((1 << (hl + 1)) - 1);
		}
		else {
			return (1 + countNodes(root->left) + countNodes(root->right));
		}		
	}
};

class Solution150 {
public:
	int evalRPN(vector<string>& tokens) {
		if (tokens.empty()) {
			return 0;
		}
		stack<int> ss;
		int i = 0;
		while (i < tokens.size()) {
			if (tokens[i][0] >= '0' && tokens[i][0] <= '9' || tokens[i].size() > 1) {
				ss.push(atoi(tokens[i].c_str()));
			}
			else {				
				int a = ss.top();
				ss.pop();
				int b = ss.top();
				ss.pop();
				switch (tokens[i][0])
				{
				case '+':
					ss.push(b + a);
					break;
				case '-':
					ss.push(b - a);
					break;
				case '*':
					ss.push(b * a);
					break;
				case '/':
					ss.push(b / a);
					break;
				default:
					break;
				}
			}
			i++;
		}

		return ss.top();
	}
};

class Solution49 {
public:
	vector<vector<string>> groupAnagrams(vector<string>& strs) {		
		vector<vector<string>> ans;
		map<string, int> m;
		int cnt = 0;
		for (int i = 0; i < strs.size(); i++) {
			string t = strs[i];
			sort(t.begin(), t.end());
			auto r = m.find(t);
			if (r == m.end()) {
				m.insert(make_pair(t, cnt));
				cnt++;
				vector<string> vs;
				vs.push_back(strs[i]);
				ans.push_back(vs);
			}
			else {
				ans[(*r).second].push_back(strs[i]);
			}
		}
		for (int i = 0; i < ans.size(); i++) {
			sort(ans[i].begin(), ans[i].end());
		}

		return ans;
	}
};

class Solution86 {
public:
	ListNode* partition(ListNode* head, int x) {
		ListNode l1(0), l2(0);		
		ListNode *p = head;
		ListNode *q1 = &l1, *q2 = &l2;
		while (p != NULL) {
			if (p->val < x) {
				q1->next = p;
				p = p->next;
				q1 = q1->next;
				q1->next = NULL;
			}
			else {
				q2->next = p;
				p = p->next;
				q2 = q2->next;
				q2->next = NULL;
			}			
		}
		q1->next = l2.next;
		return l1.next;
	}
};

class Solution114 {
public:
	void flatten(TreeNode* root) {
		if (root == NULL) {
			return;
		}
		flatten(root->left);
		flatten(root->right);
		TreeNode *r = root->right;
		root->right = root->left;
		root->left = NULL;
		TreeNode *p = root;
		while (p->right != NULL) {
			p = p->right;
		}
		p->right = r;
	}
};

class Solution199 {
public:
	vector<int> rightSideView(TreeNode* root) {
		vector<int> ans;
		if (root == NULL) {
			return ans;
		}		
		vector<TreeNode*> vt;
		vt.push_back(root);
		while (!vt.empty()) {
			ans.push_back(vt[vt.size() - 1]->val);
			vector<TreeNode*> vt2;			
			for (int i = 0; i < vt.size(); i++) {
				if (vt[i]->left != NULL) {
					vt2.push_back(vt[i]->left);
				}
				if (vt[i]->right != NULL) {
					vt2.push_back(vt[i]->right);
				}
			}
			vt = vt2;
		}

		return ans;
	}
};

class Solution98 {
public:
	bool isValidBST(TreeNode* root) {
		int nBit = sizeof(long long) * 8;
		long long mm = ~(1 << 31);
		return isValidBSTr(root, -mm - 2, mm + 1);
	}
private:
	bool isValidBSTr(TreeNode *root, long long m1, long long m2) {
		if (m1 >= m2) {
			return false;
		}
		if (root == NULL) {
			return true;
		}
		if (!(m1 < root->val && root->val < m2)) {
			return false;
		}
		return (isValidBSTr(root->left, m1, min((long long)root->val, m2)) && isValidBSTr(root->right, max((long long)root->val, m1), m2));
	}
};

class Solution43 {
public:
	string multiply(string num1, string num2) {
		strToDigit(num1);
		strToDigit(num2);
		if (num1.size() == 0 || num2.size() == 0) {
			return "0";
		}		
		string ans;
		int n = num1.size() + num2.size();
		ans.resize(n, 0);
		int c = 0;
		for (int i = 0; i < num2.size(); i++) {
			int j = 0;
			for ( ; j < num1.size(); j++) {
				ans[i + j] += num2[i] * num1[j] + c;
				c = ans[i + j] / 10;
				ans[i + j] %= 10;
			}
			if (c != 0) {
				ans[i + j] = c;
				c = 0;
			}
		}		
		if (ans[n - 1] == 0) {
			ans.erase(n - 1, 1);
		}
		reverse(ans.begin(), ans.end());
		for (int i = 0; i < ans.size(); i++) {
			ans[i] += '0';
		}
		return ans;
	}

private:
	void strToDigit(string &num) {
		int cnt = 0;
		while (cnt < num.size() && num[cnt] == '0') {
			cnt++;
		}
		num.erase(0, cnt);
		for (int i = 0; i < num.size(); i++) {
			num[i] -= '0';
		}
		reverse(num.begin(), num.end());
	}
};

class Solution169 {
public:
	int majorityElement(vector<int>& nums) {
		int a, b;
		int na = 0, nb = 0;
		for (int i = 0; i < nums.size(); i++) {
			if (na == 0) {
				na++;
				a = nums[i];
			}
			else if (nb == 0) {
				nb++;
				b = nums[i];
			}
			if (na != 0 && nb != 0) {
				if (a == b) {
					na++;
					nb--;
				}
				else {
					na--;
					nb--;
				}
			}
		}

		if (na != 0) {
			return a;
		}
		else {
			return b;
		}
	}
};

class Solution169_2 {
public:
	int majorityElement(vector<int>& nums) {
		int a;
		int na = 0;
		int *ptr = nums.data();
		for (int i = 0; i < nums.size(); i++) {
			if (na == 0) {
				na++;
				a = ptr[i];
			}
			else if (a == ptr[i]) {
				na++;
			}
			else {
				na--;
			}
		}

		return a;
	}
};

class Solution229 {
public:
	vector<int> majorityElement(vector<int>& nums) {
		int a, b;
		int na = 0, nb = 0, nc = 0;
		for (int i = 0; i < nums.size(); i++) {
			if (na == 0) {
				na++;
				a = nums[i];
			}
			else if (nums[i] == a) {
				na++;
			}
			else if (nb == 0) {
				nb++;
				b = nums[i];
			}			
			else if (nums[i] == b) {
				nb++;
			}
			else {
				na--;
				nb--;
				if (na == 0 && nb != 0) {
					a = b;
					na = nb;
					nb = 0;
				}
				nc++;
			}
		}

		na = 0;
		nb = 0;
		for (int i = 0; i < nums.size(); i++) {
			if (nums[i] == a) {
				na++;
			}
			else if (nums[i] == b) {
				nb++;
			}
		}
		vector<int> ans;
		if (na > nums.size() / 3) {
			ans.push_back(a);
		}
		if (nb > nums.size() / 3) {
			ans.push_back(b);
		}

		return ans;
	}
};

//Attention!!!
class Solution134 {
public:
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
		vector<int> rem(gas.size());
		for (int i = 0; i < gas.size(); i++) {
			rem[i] = gas[i] - cost[i];
		}
		
		int left = 0, start = 0, sum = 0;
		for (int i = 0; i < rem.size(); i++) {
			left += rem[i];
			sum += rem[i];
			if (sum < 0) {
				start = i + 1;
				sum = 0;
			}
		}

		if (left < 0) {
			return -1;
		}
		return start;
	}
};

class Solution47 {
public:
	vector<vector<int>> permuteUnique(vector<int>& nums) {		
		vector<vector<int>> ans;
		sort(nums.begin(), nums.end());
		if (nums.size() == 0) {
			return ans;
		}
		bool hasNext = false;
		do {
			ans.push_back(nums);
			hasNext = next_permutation(nums.begin(), nums.end());
		} while (hasNext);

		return ans;
	}
};

//Attention!!!
class TrieNode {
public:
	// Initialize your data structure here.
	TrieNode() {
		isKey = false;
		for (int i = 0; i < 26; i++) {
			children[i] = NULL;
		}
	}

	bool isKey;
	TrieNode *children[26];
};

class Trie {
public:
	Trie() {
		root = new TrieNode();		
	}

	// Inserts a word into the trie.
	void insert(string word) {
		if (word.size() == 0) {
			return;
		}
		TrieNode *tn = root;
		for (int i = 0; i < word.size(); i++) {
			int k = word[i] - 'a';
			if (tn->children[k] == NULL)  {
				tn->children[k] = new TrieNode;
			}
			tn = tn->children[k];
		}
		tn->isKey = true;
	}

	// Returns if the word is in the trie.
	bool search(string word) {
		if (word.size() == 0) {
			return false;
		}
		TrieNode *tn = root;
		for (int i = 0; i < word.size(); i++) {
			int k = word[i] - 'a';
			if (tn->children[k] == NULL) {
				return false;
			}
			tn = tn->children[k];
		}
		return tn->isKey;
	}

	// Returns if there is any word in the trie
	// that starts with the given prefix.
	bool startsWith(string prefix) {
		TrieNode *tn = root;
		for (int i = 0; i < prefix.size(); i++) {
			int k = prefix[i] - 'a';
			if (tn->children[k] == NULL) {
				return false;
			}
			tn = tn->children[k];
		}
		return true;
	}

private:
	TrieNode* root;	
};

class WordDictionary {
public:
	WordDictionary() {
		root = new TrieNode;
	}

	WordDictionary(TrieNode *t) {
		root = t;
	}

	// Adds a word into the data structure.
	void addWord(string word) {
		if (word.size() == 0) {
			return;
		}
		TrieNode *tn = root;
		for (int i = 0; i < word.size(); i++) {
			int k = word[i] - 'a';
			if (tn->children[k] == NULL)  {
				tn->children[k] = new TrieNode;
			}
			tn = tn->children[k];
		}
		tn->isKey = true;
	}

	// Returns if the word is in the data structure. A word could
	// contain the dot character '.' to represent any one letter.
	bool search(string word) {
		TrieNode *tn = root;
		for (int i = 0; i < word.size(); i++) {
			int k = word[i] - 'a';
			if (word[i] != '.') {				
				if (tn->children[k] == NULL)  {
					return false;
				}
				tn = tn->children[k];
			}
			else {
				string s = string(word, i + 1, word.size() - 1);
				for (int i = 0; i < 26; i++) {					
					if (tn->children[i] != NULL) {
						WordDictionary d(tn->children[i]);						
						if (d.search(s)) {
							return true;
						}
					}
				}
				return false;
			}
		}
		return tn->isKey;
	}

private:
	TrieNode *root;
};

class Solution39 {
public:
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		sort(candidates.begin(), candidates.end());		
		return combinationSum(candidates, 0, target);
	}

	vector<vector<int>> combinationSum(vector<int>& candidates, int start, int target) {
		vector<vector<int>> ans;		
		if (target == 0) {
			vector<int> v;
			ans.push_back(v);
		}
		for (int i = start; i < candidates.size() && candidates[i] <= target; i++) {
			vector<int> v;
			v.push_back(candidates[i]);
			vector<vector<int>> r = combinationSum(candidates, i, target - candidates[i]);
			for (int j = 0; j < r.size(); j++) {
				vector<int> t;
				t.push_back(candidates[i]);
				for (int k = 0; k < r[j].size(); k++) {
					t.push_back(r[j][k]);
				}
				ans.push_back(t);
			}
		}
		return ans;
	}
};

class Solution40 {
public:
	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
		sort(candidates.begin(), candidates.end());
		return combinationSum2(candidates, 0, target);
	}

	vector<vector<int>> combinationSum2(vector<int>& candidates, int start, int target) {
		vector<vector<int>> ans;
		if (target == 0) {
			vector<int> v;
			ans.push_back(v);
		}
		for (int i = start; i < candidates.size() && candidates[i] <= target; i++) {
			if (i > start && candidates[i] == candidates[i - 1]) {
				continue;
			}
			vector<int> v;
			v.push_back(candidates[i]);
			vector<vector<int>> r = combinationSum2(candidates, i + 1, target - candidates[i]);
			for (int j = 0; j < r.size(); j++) {
				vector<int> t;
				t.push_back(candidates[i]);
				for (int k = 0; k < r[j].size(); k++) {
					t.push_back(r[j][k]);
				}
				ans.push_back(t);
			}
			
		}
		return ans;
	}
};

class Solution216 {
public:
	vector<vector<int>> combinationSum3(int k, int n) {
		return combinationSum3(k, 1, n);
	}

	vector<vector<int>> combinationSum3(int k, int start, int n) {
		vector<vector<int>> ans;
		if (k == 0) {
			if (n == 0) {
				vector<int> v;
				ans.push_back(v);
			}
			return ans;
		}
		for (int i = start; i <= 9; i++) {
			vector<vector<int>> r = combinationSum3(k - 1, i + 1, n - i);
			for (int j = 0; j < r.size(); j++) {
				vector<int> t;
				t.push_back(i);
				for (int l = 0; l < r[j].size(); l++) {
					t.push_back(r[j][l]);
				}
				ans.push_back(t);
			}
		}
		return ans;
	}
};

class Solution209 {
public:
	int minSubArrayLen(int s, vector<int>& nums) {
		int minLen = ~(1 << 31);
		for (int i = 0; i < nums.size(); i++) {
			int sum = nums[i];
			int j = i + 1;
			while (j < nums.size() && sum < s && j - i < minLen) {
				sum += nums[j];
				j++;
			}
			if (sum >= s && j - i < minLen) {
				minLen = j - i;
			}
		}

		if (minLen > nums.size()) {
			return 0;
		}
		return minLen;
	}
};

class Solution69 {
public:
	int mySqrt(int x) {
		if (x <= 1) {
			return x;
		}
		double last = 0;
		double res = 1;
		while (abs(last - res) > 0.9) {
			last = res;
			res = (res + x / res) / 2;
		}

		return (int)res;
	}
};

class Solution139 {
public:
	bool wordBreak(string s, unordered_set<string>& wordDict) {
		int maxLen = 0;
		for (auto it = wordDict.begin(); it != wordDict.end(); it++) {
			if ((*it).size() > maxLen) {
				maxLen = (*it).size();
			}
		}
		vector<bool> vb(s.size() + 1);
		vb[0] = true;
		for (int i = 0; i < s.size(); i++) {
			for (int j = max(0, i - maxLen + 1); j <= i; j++) {
				string t = string(s, j, i - j + 1);
				if (vb[j] && wordDict.find(t) != wordDict.end()) {
					vb[i + 1] = true;
					break;
				}
			}
		}

		return vb[s.size()];
	}
};

//Attention!!!
class Solution140 {
public:
	vector<string> wordBreak(string s, unordered_set<string>& wordDict) {
		int maxLen = 0;
		for (auto it = wordDict.begin(); it != wordDict.end(); it++) {
			if ((*it).size() > maxLen) {
				maxLen = (*it).size();
			}
		}
		vector<bool> vb(s.size() + 1);
		vb[0] = true;
		for (int i = 0; i < s.size(); i++) {
			for (int j = max(0, i - maxLen + 1); j <= i; j++) {
				string t = string(s, j, i - j + 1);
				if (vb[j] && wordDict.find(t) != wordDict.end()) {
					vb[i + 1] = true;
					break;
				}
			}
		}

		vector<string> ans;
		if (vb[s.size()]) {
			ans = wordBreak(s, 0, wordDict, vb);
		}		

		return ans;
	}

	vector<string> wordBreak(string &s, int start, unordered_set<string>& wordDict, vector<bool> &vb) {
		vector<string> vs;
		if (start >= s.size()) {
			vs.push_back("");
			return vs;
		}
		string a;
		for (int i = start; i < s.size(); i++) {
			a += s[i];
			if (vb[i + 1] && wordDict.find(a) != wordDict.end()) {
				vector<string> vs2 = wordBreak(s, i + 1, wordDict, vb);
				for (int j = 0; j < vs2.size(); j++) {
					if (!vs2[j].empty()) {
						vs.push_back(a + " " + vs2[j]);
					}
					else {
						vs.push_back(a);
					}
				}
			}
		}
		return vs;
	}
};

class Solution166 {
public:
	string fractionToDecimal(int numerator1, int denominator1) {
		string ans;
		if (denominator1 == 0) {
			return ans;
		}
		if (numerator1 == 0) {
			return "0";
		}
		long long numerator = numerator1;
		long long denominator = denominator1;	
		bool nNeg = false, dNeg = false;
		if (numerator < 0) {
			nNeg = true;
			numerator *= -1;
		}
		if (denominator < 0) {
			dNeg = true;
			denominator *= -1;
		}
		if (nNeg == true && dNeg == false || nNeg == false && dNeg == true) {
			ans += '-';
		}
		ans += i2a(numerator / denominator);		
		numerator %= denominator;
		if (numerator != 0) {
			ans += '.';
			map<int, int> m;
			while (numerator != 0 && m.find(numerator) == m.end())  {
				m.insert(make_pair(numerator, ans.size()));
				numerator *= 10;
				long long t = numerator / denominator;
				ans += (char)(t + '0');
				numerator %= denominator;
			}
			if (numerator != 0) {
				auto a = m.find(numerator);
				ans.insert((*a).second, 1, '(');
				ans += ')';
			}
		}
		
		return ans;
	}

private:
	string i2a(long long a) {
		bool neg = false;
		if (a < 0) {
			neg = true;
			a *= -1;
		}
		string s;
		while (a != 0) {
			s += (char)(a % 10 + '0');
			a /= 10;
		}
		if (neg) {
			s += '-';
		}
		reverse(s.begin(), s.end());
		if (s.empty()) {
			s += '0';
		}
		return s;
	}
};

class Solution130 {
public:
	void solve(vector<vector<char>>& board) {
		if (board.size() == 0 || board[0].size() == 0) {
			return;
		}
		int m = board.size(), n = board[0].size();
		set<pair<int, int>> ts;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				set<pair<int, int>> s;
				if (board[i][j] == 'O' && ts.find(make_pair(i, j)) == ts.end() && !hasMargin(board, i, j, s)) {
					for (auto it = s.begin(); it != s.end(); it++) {
						board[(*it).first][(*it).second] = 'X';
					}
				}
				else {
					for (auto it = s.begin(); it != s.end(); it++) {
						ts.insert(*it);
					}
				}
			}
		}
	}

private:
	bool hasMargin(vector<vector<char>>& board, int _i, int _j, set<pair<int, int>> &s) {
		queue<pair<int, int>> q;
		q.push(make_pair(_i, _j));
		while (!q.empty()) {
			pair<int, int> p = q.front();
			q.pop();
			int i = p.first, j = p.second;
			if (board[i][j] == 'X') {
				continue;
			}
			if (s.find(make_pair(i, j)) == s.end()) {
				s.insert(make_pair(i, j));
				if (0 == i || board.size() - 1 == i || 0 == j || board[0].size() - 1 == j) {
					return true;
				}
				q.push(make_pair(i - 1, j));
				q.push(make_pair(i + 1, j));
				q.push(make_pair(i, j - 1));
				q.push(make_pair(i, j + 1));
			}
		}

		return false;
	}
};

class Solution130_2 {
public:
	void solve(vector<vector<char>>& board) {
		if (board.size() == 0 || board[0].size() == 0) {
			return;
		}
		queue<pair<int, int>> q;
		int m = board.size(), n = board[0].size();
		for (int i = 0; i < n; i++) {
			if (board[0][i] == 'O') {
				q.push(make_pair(0, i));
			}
			if (board[m - 1][i] == 'O') {
				q.push(make_pair(m - 1, i));
			}
		}
		for (int i = 0; i < m; i++) {
			if (board[i][0] == 'O') {
				q.push(make_pair(i, 0));
			}
			if (board[i][n - 1] == 'O') {
				q.push(make_pair(i, n - 1));
			}
		}

		while (!q.empty()) {
			pair<int, int> p = q.front();
			q.pop();
			int i = p.first, j = p.second;
			board[i][j] = 'M';
			if (i - 1 >= 0 && board[i - 1][j] == 'O') {
				q.push(make_pair(i - 1, j));
			}
			if (i + 1 < m && board[i + 1][j] == 'O') {
				q.push(make_pair(i + 1, j));
			}
			if (j - 1 >= 0 && board[i][j - 1] == 'O') {
				q.push(make_pair(i, j - 1));
			}
			if (j + 1 < n && board[i][j + 1] == 'O') {
				q.push(make_pair(i, j + 1));
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O') {
					board[i][j] = 'X';
				}
				else if (board[i][j] == 'M') {
					board[i][j] = 'O';
				}
			}
		}
	}
};

class Solution109 {
public:
	TreeNode* sortedListToBST(ListNode* head) {
		ListNode ln(0);
		ln.next = head;
		int cnt = 0;
		ListNode *q = head;
		while (q != NULL) {
			cnt++;
			q = q->next;
		}
		return sortedListToBST(&ln, cnt);
	}

	TreeNode* sortedListToBST(ListNode* ln, int n) {
		if (n <= 0) {
			return NULL;
		}
		ListNode *q = ln;
		for (int i = 0; i < n / 2; i++) {
			q = q->next;
		}
		TreeNode *tr = new TreeNode(q->next->val);
		tr->left = sortedListToBST(ln, n / 2);
		tr->right = sortedListToBST(q->next, n - n / 2 - 1);
		return tr;
	}
};

class Iterator {
	struct Data;
	Data* data;
public:
	Iterator(const vector<int>& nums);
	Iterator(const Iterator& iter);
	virtual ~Iterator();
	// Returns the next element in the iteration.
	int next();
	// Returns true if the iteration has more elements.
	bool hasNext() const;
};

class PeekingIterator : public Iterator {
public:
	PeekingIterator(const vector<int>& nums) : Iterator(nums) {
		// Initialize any member here.
		// **DO NOT** save a copy of nums and manipulate it directly.
		// You should only use the Iterator interface methods.
		nCnt = nums.size();
		nThis = 0;
		pv = &nums;
	}

	// Returns the next element in the iteration without advancing the iterator.
	int peek() {
		return (*pv)[nThis];
	}

	// hasNext() and next() should behave the same as in the Iterator interface.
	// Override them if needed.
	int next() {
		return (*pv)[nThis++];
	}

	bool hasNext() const {
		return (nThis < nCnt);
	}

private:
	int nCnt;
	int nThis;
	const vector<int> *pv;
};

class BSTIterator {
public:
	BSTIterator(TreeNode *root) {
		TreeNode *p = root;
		while (p) {
			s.push(p);
			p = p->left;
		}
	}

	/** @return whether we have a next smallest number */
	bool hasNext() {
		return !s.empty();
	}

	/** @return the next smallest number */
	int next() {
		TreeNode *r = s.top();
		s.pop();
		TreeNode *p = r->right;
		while (p) {
			s.push(p);
			p = p->left;
		}
		return r->val;
	}

private:
	stack<TreeNode *> s;
};

class Solution63 {
public:
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
		if (obstacleGrid.size() <= 0 || obstacleGrid[0].size() <= 0) {
			return 0;
		}
		int m = obstacleGrid.size(), n = obstacleGrid[0].size();
		vector<int> v(n);
		if (obstacleGrid[0][0] == 1) {
			return 0;
		}
		v[0] = 1;
		for (int j = 1; j < n; j++) {
			if (obstacleGrid[0][j] == 1) {
				v[j] = 0;
			}
			else {
				v[j] = v[j - 1];
			}
		}
		bool hasOb = false;
		for (int i = 1; i < m; i++) {
			if (obstacleGrid[i][0] == 1) {
				hasOb = true;
			}
			if (hasOb) {
				v[0] = 0;
			}
			for (int j = 1; j < n; j++) {
				if (obstacleGrid[i][j] != 1) {
					v[j] = v[j - 1] + v[j];					
				}
				else {
					v[j] = 0;
				}
			}
		}

		return v[n - 1];
	}
};

class Solution120 {
public:
	int minimumTotal(vector<vector<int>>& triangle) {
		if (triangle.size() == 0) {
			return 0;
		}
		for (int i = 1; i < triangle.size(); i++) {
			triangle[i][0] += triangle[i - 1][0];
			int j;
			for (j = 1; j < triangle[i].size() - 1; j++) {
				triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j]);
			}
			triangle[i][j] += triangle[i - 1][j - 1];
		}
		int mm = ~(1 << 31);
		int n = triangle.size() - 1;
		for (int i = 0; i < triangle[n].size(); i++) {
			mm = min(mm, triangle[n][i]);
		}

		return mm;
	}
};

//Attention!!!
class Solution213 {
public:
	int rob(vector<int>& nums) {
		if (nums.size() == 1) {
			return nums[0];
		}
		return max(rob(nums, 0, nums.size() - 1), rob(nums, 1, nums.size()));
	}

	int rob(vector<int>& nums, int start, int end) {
		int a = 0, b = 0;
		for (int i = start; i < end; i++) {
			int t = max(a + nums[i], b);
			a = b;
			b = t;
		}
		return b;
	}
};

class Solution34 {
public:
	vector<int> searchRange(vector<int>& nums, int target) {
		vector<int> notFound = { -1, -1 };
		vector<int> ans;
		int low = 0, high = nums.size() - 1;
		while (low <= high) {
			int mid = (low + high) / 2;
			if (nums[mid] >= target) {
				high = mid - 1;
			}
			else {
				low = mid + 1;
			}
		}
		ans.push_back(low);
		low = 0;
		high = nums.size() - 1;
		while (low <= high) {
			int mid = (low + high) / 2;
			if (nums[mid] <= target) {
				low = mid + 1;
			}
			else {
				high = mid - 1;
			}
		}
		ans.push_back(high);
		if (ans[1] >= ans[0]) {
			return ans;
		}
		return notFound;
	}
};

class Solution103 {
public:
	vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
		vector<vector<int>> ans;
		if (root == NULL) {
			return ans;
		}
		vector<TreeNode *> vt;
		vt.push_back(root);
		while (!vt.empty()) {
			vector<TreeNode *> vt2;
			vector<int> vi;
			for (int i = 0; i < vt.size(); i++) {
				vi.push_back(vt[i]->val);
				if (vt[i]->left) {
					vt2.push_back(vt[i]->left);
				}
				if (vt[i]->right) {
					vt2.push_back(vt[i]->right);
				}
			}
			ans.push_back(vi);
			vt = vt2;
		}
		for (int i = 0; i < ans.size(); i++) {
			if (i & 1) {
				reverse(ans[i].begin(), ans[i].end());
			}
		}

		return ans;
	}
};

class Solution240 {
public:
	bool searchMatrix(vector<vector<int>>& matrix, int target) {
		return searchMatrix(matrix, target, 0, 0, matrix.size() - 1, matrix[0].size() - 1);
	}

	bool searchMatrix(vector<vector<int>>& matrix, int target, int i1, int j1, int i2, int j2) {
		if (i1 > i2 || j1 > j2) {
			return false;
		}
		int im = (i1 + i2) / 2;
		int jm = (j1 + j2) / 2;
		if (matrix[im][jm] == target) {
			return true;
		}
		else if (matrix[im][jm] > target) {
			return searchMatrix(matrix, target, i1, j1, im - 1, j2) || searchMatrix(matrix, target, im, j1, i2, jm - 1);
		}
		else {
			return searchMatrix(matrix, target, im + 1, j1, i2, j2) || searchMatrix(matrix, target, i1, jm + 1, im, j2);
		}
	}
};

class Solution131 {
public:
	vector<vector<string>> partition(string &s) {
		vector<vector<string>> vvs;
		if (isPalindrome(s)) {
			vector<string> vs = { s };
			vvs.push_back(vs);
		}
		for (int i = s.size() - 1; i > 0; i--) {
			vector<string> vs;
			vs.push_back(string(s, 0, i));
			vs.push_back(string(s, i, s.size() - i));
			if (isPalindrome(vs[1])) {				
				vector<vector<string>> vvs2 = partition(vs[0]);
				for (int j = 0; j < vvs2.size(); j++) {
					vvs2[j].push_back(vs[1]);
					vvs.push_back(vvs2[j]);
				}
			}
		}
		
		return vvs;
	}
	
	bool isPalindrome(string &s) {
		for (int i = 0; i < s.size() / 2; i++) {
			if (s[i] != s[s.size() - 1 - i]) {
				return false;
			}
		}
		return true;
	}
};

class Solution131_2 {
public:
	vector<vector<string>> partition(string &s) {
		vector<vector<string>> vvs;
		if (isPalindrome(s)) {
			vector<string> vs = { s };
			vvs.push_back(vs);
		}
		for (int i = 1; i < s.size(); i++) {
			vector<string> vs;
			vs.push_back(string(s, 0, i));
			vs.push_back(string(s, i, s.size() - i));
			if (isPalindrome(vs[0])) {
				vector<vector<string>> vvs2 = partition(vs[1]);
				for (int j = 0; j < vvs2.size(); j++) {
					vector<string> vs1 = {vs[0]};
					for (int k = 0; k < vvs2[j].size(); k++) {						
						vs1.push_back(vvs2[j][k]);
					}					
					vvs.push_back(vs1);
				}
			}
		}

		return vvs;
	}

	bool isPalindrome(string &s) {
		for (int i = 0; i < s.size() / 2; i++) {
			if (s[i] != s[s.size() - 1 - i]) {
				return false;
			}
		}
		return true;
	}
};

class Solution139_3 {
public:
	vector<vector<string>> partition(string &s) {		
		dfs(s, 0);
		return vvs;
	}

	void dfs(string &s, int k) {
		if (k >= s.size()) {
			vvs.push_back(vs);
			return;
		}
		string t;
		for (int i = k; i < s.size(); i++) {
			t += s[i];
			if (isPalindrome(t)) {
				vs.push_back(t);
				dfs(s, i + 1);
				vs.pop_back();
			}
		}
	}

private:
	vector<vector<string>> vvs;
	vector<string> vs;
	bool isPalindrome(string &s) {
		for (int i = 0; i < s.size() / 2; i++) {
			if (s[i] != s[s.size() - 1 - i]) {
				return false;
			}
		}
		return true;
	}
};

class Solution187 {
public:
	vector<string> findRepeatedDnaSequences(string s) {
		vector<string> vs;
		if (s.size() < 10) {
			return vs;
		}
		unsigned char *table = new unsigned char[1 << 20];
		for (int i = 0; i < (1 << 20); i++) {
			table[i] = 0;
		}
		int t = 0;
		for (int i = 0; i < 10; i++) {
			t = (t << 2) | encode(s[i]);
		}
		table[t]++;
		for (int i = 10; i < s.size(); i++) {
			t = ((t << 2) & 0xfffff) | encode(s[i]);
			table[t]++;
		}
		for (int i = 0; i < (1 << 20); i++) {
			if (table[i] > 1) {
				vs.push_back(decode(i));
			}
		}
		delete[] table;
		return vs;
	}

private:
	inline int encode(char c) {
		switch (c) {
		case 'A':
			return 0;
		case 'C':
			return 1;
		case 'G':
			return 2;
		case 'T':
			return 3;
		default:
			return 0;
		}
	}

	string decode(int t) {
		string s;
		for (int i = 0; i < 10; i++) {
			int r = t & 3;
			t >>= 2;
			switch (r)
			{
			case 0:
				s += 'A'; break;
			case 1:
				s += 'C'; break;
			case 2:
				s += 'G'; break;
			case 3:
				s += 'T'; break;
			default:
				break;
			}
		}
		reverse(s.begin(), s.end());
		return s;
	}
};

class Solution187_2 {
public:
	vector<string> findRepeatedDnaSequences(string s) {
		vector<string> vs;
		if (s.size() < 10) {
			return vs;
		}
		unordered_map<int, int> um;
		int t = 0;
		for (int i = 0; i < 10; i++) {
			t = (t << 3) | (s[i] & 7);
		}
		um[t]++;
		int ss = s.size();
		for (int i = 10; i < ss; i++) {
			t = ((t << 3) & (~(3 << 30))) | (s[i] & 7);
			um[t]++;
			if (um[t] == 2) {
				vs.push_back(string(s, i - 9, 10));
			}
		}

		return vs;
	}
};

class Solution221 {
public:
	int maximalSquare(vector<vector<char>>& matrix) {
		if (matrix.size() == 0 || matrix[0].size() == 0) {
			return 0;
		}
		int m = matrix.size(), n = matrix[0].size();
		vector<vector<int>> con;
		for (int i = 0; i < m; i++) {
			vector<int> v(n);
			int cnt = 0;
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == '1') {
					v[j] = ++cnt;
				}
				else {
					cnt = 0;
					v[j] = 0;
				}
			}
			con.push_back(v);
		}
		for (int j = 0; j < n; j++) {
			int cnt = 0;
			for (int i = 0; i < m; i++) {
				if (matrix[i][j] == '1') {
					con[i][j] = min(++cnt, con[i][j]);
				}
				else {
					cnt = 0;
					con[i][j] = 0;
				}
			}
		}

		int maxSquare = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int k;
				for (k = 0; i + k < m && j + k < n; k++) {
					if (con[i + k][j + k] <= k) {
						break;
					}
				}
				if (k * k > maxSquare) {
					maxSquare = k * k;
				}
			}
		}

		return maxSquare;
	}
};

class Solution {
public:
	vector<vector<string>> findLadders(string beginWord, string endWord, unordered_set<string>& wordList) {
		wordList.insert(beginWord);
		wordList.insert(endWord);
		vector<vector<pair<string, vector<int>>>> vv;
		vv.push_back({ make_pair(beginWord, vector<int>{})});
		wordList.erase(beginWord);
		vector<string> ve;
		int i = 0;
		while (1) {			
			unordered_multimap<string, vector<int>> um;
			for (int j = 0; j < vv[i].size(); j++) {
				string b = vv[i][j].first;		
				if (b == endWord) {
					um.clear();
					break;
				}
				for (int k = 0; k < b.size(); k++) {
					char t = b[k];
					for (int l = 'a'; l <= 'z'; l++) {
						if (l != t) {
							b[k] = l;
							auto it = wordList.find(b);
							if (it != wordList.end()) {
								auto it2 = um.find(b);
								if (it2 != um.end()) {
									(*it2).second.push_back(j);
								}
								else {
									um.insert(make_pair(b, vector<int>{j}));
								}
								ve.push_back(b);
							}
							b[k] = t;
						}
					}
				}
			}

			vector<pair<string, vector<int>>> vp;
			for (auto it = um.begin(); it != um.end(); it++) {
				vp.push_back(*it);
			}
			if (vp.empty()) {
				break;
			}
			vv.push_back(vp);
			for (int j = 0; j < ve.size(); j++) {
				wordList.erase(ve[j]);
			}
			ve.clear();
			i++;
		}

		int n = vv.size() - 1;
		for (int i = 0; i < vv[n].size(); i++) {
			if (vv[n][i].first == endWord) {
				getVVS(vv, n, i);
			}
		}

		return vvs;
	}

private:
	vector<vector<string>> vvs;
	vector<string> vs;	
	void getVVS(vector<vector<pair<string, vector<int>>>>& vv, int n, int i) {		
		pair<string, vector<int>> &p = vv[n][i];
		vs.push_back(p.first);
		if (n == 0) {
			vector<string> vs1;
			for (int j = vs.size() - 1; j >= 0; j--) {
				vs1.push_back(vs[j]);
			}
			vvs.push_back(vs1);
		}
		for (int j = 0; j < p.second.size(); j++) {
			getVVS(vv, n - 1, p.second[j]);
		}
		vs.pop_back();
	}
};

int main()
{
	Timer timer;
	Solution s;
	vector<vector<char>> vv = { {'1'} };
	//vector<vector<int>> vv = { { 2 }, { 3, 4 }, { 6, 5, 7 }, {4, 1, 8, 3} };
	vector<int> vi = { 1, 2, 2, 2, 4 };
	unordered_set<string> us = { "slit", "bunk", "wars", "ping", "viva", "wynn", "wows", "irks", "gang", "pool", "mock", "fort", "heel", "send", "ship", "cols", "alec", "foal", "nabs", "gaze", "giza", "mays", "dogs", "karo", "cums", "jedi", "webb", "lend", "mire", "jose", "catt", "grow", "toss", "magi", "leis", "bead", "kara", "hoof", "than", "ires", "baas", "vein", "kari", "riga", "oars", "gags", "thug", "yawn", "wive", "view", "germ", "flab", "july", "tuck", "rory", "bean", "feed", "rhee", "jeez", "gobs", "lath", "desk", "yoko", "cute", "zeus", "thus", "dims", "link", "dirt", "mara", "disc", "limy", "lewd", "maud", "duly", "elsa", "hart", "rays", "rues", "camp", "lack", "okra", "tome", "math", "plug", "monk", "orly", "friz", "hogs", "yoda", "poop", "tick", "plod", "cloy", "pees", "imps", "lead", "pope", "mall", "frey", "been", "plea", "poll", "male", "teak", "soho", "glob", "bell", "mary", "hail", "scan", "yips", "like", "mull", "kory", "odor", "byte", "kaye", "word", "honk", "asks", "slid", "hopi", "toke", "gore", "flew", "tins", "mown", "oise", "hall", "vega", "sing", "fool", "boat", "bobs", "lain", "soft", "hard", "rots", "sees", "apex", "chan", "told", "woos", "unit", "scow", "gilt", "beef", "jars", "tyre", "imus", "neon", "soap", "dabs", "rein", "ovid", "hose", "husk", "loll", "asia", "cope", "tail", "hazy", "clad", "lash", "sags", "moll", "eddy", "fuel", "lift", "flog", "land", "sigh", "saks", "sail", "hook", "visa", "tier", "maws", "roeg", "gila", "eyes", "noah", "hypo", "tore", "eggs", "rove", "chap", "room", "wait", "lurk", "race", "host", "dada", "lola", "gabs", "sobs", "joel", "keck", "axed", "mead", "gust", "laid", "ends", "oort", "nose", "peer", "kept", "abet", "iran", "mick", "dead", "hags", "tens", "gown", "sick", "odis", "miro", "bill", "fawn", "sumo", "kilt", "huge", "ores", "oran", "flag", "tost", "seth", "sift", "poet", "reds", "pips", "cape", "togo", "wale", "limn", "toll", "ploy", "inns", "snag", "hoes", "jerk", "flux", "fido", "zane", "arab", "gamy", "raze", "lank", "hurt", "rail", "hind", "hoot", "dogy", "away", "pest", "hoed", "pose", "lose", "pole", "alva", "dino", "kind", "clan", "dips", "soup", "veto", "edna", "damp", "gush", "amen", "wits", "pubs", "fuzz", "cash", "pine", "trod", "gunk", "nude", "lost", "rite", "cory", "walt", "mica", "cart", "avow", "wind", "book", "leon", "life", "bang", "draw", "leek", "skis", "dram", "ripe", "mine", "urea", "tiff", "over", "gale", "weir", "defy", "norm", "tull", "whiz", "gill", "ward", "crag", "when", "mill", "firs", "sans", "flue", "reid", "ekes", "jain", "mutt", "hems", "laps", "piss", "pall", "rowe", "prey", "cull", "knew", "size", "wets", "hurl", "wont", "suva", "girt", "prys", "prow", "warn", "naps", "gong", "thru", "livy", "boar", "sade", "amok", "vice", "slat", "emir", "jade", "karl", "loyd", "cerf", "bess", "loss", "rums", "lats", "bode", "subs", "muss", "maim", "kits", "thin", "york", "punt", "gays", "alpo", "aids", "drag", "eras", "mats", "pyre", "clot", "step", "oath", "lout", "wary", "carp", "hums", "tang", "pout", "whip", "fled", "omar", "such", "kano", "jake", "stan", "loop", "fuss", "mini", "byrd", "exit", "fizz", "lire", "emil", "prop", "noes", "awed", "gift", "soli", "sale", "gage", "orin", "slur", "limp", "saar", "arks", "mast", "gnat", "port", "into", "geed", "pave", "awls", "cent", "cunt", "full", "dint", "hank", "mate", "coin", "tars", "scud", "veer", "coax", "bops", "uris", "loom", "shod", "crib", "lids", "drys", "fish", "edit", "dick", "erna", "else", "hahs", "alga", "moho", "wire", "fora", "tums", "ruth", "bets", "duns", "mold", "mush", "swop", "ruby", "bolt", "nave", "kite", "ahem", "brad", "tern", "nips", "whew", "bait", "ooze", "gino", "yuck", "drum", "shoe", "lobe", "dusk", "cult", "paws", "anew", "dado", "nook", "half", "lams", "rich", "cato", "java", "kemp", "vain", "fees", "sham", "auks", "gish", "fire", "elam", "salt", "sour", "loth", "whit", "yogi", "shes", "scam", "yous", "lucy", "inez", "geld", "whig", "thee", "kelp", "loaf", "harm", "tomb", "ever", "airs", "page", "laud", "stun", "paid", "goop", "cobs", "judy", "grab", "doha", "crew", "item", "fogs", "tong", "blip", "vest", "bran", "wend", "bawl", "feel", "jets", "mixt", "tell", "dire", "devi", "milo", "deng", "yews", "weak", "mark", "doug", "fare", "rigs", "poke", "hies", "sian", "suez", "quip", "kens", "lass", "zips", "elva", "brat", "cosy", "teri", "hull", "spun", "russ", "pupa", "weed", "pulp", "main", "grim", "hone", "cord", "barf", "olav", "gaps", "rote", "wilt", "lars", "roll", "balm", "jana", "give", "eire", "faun", "suck", "kegs", "nita", "weer", "tush", "spry", "loge", "nays", "heir", "dope", "roar", "peep", "nags", "ates", "bane", "seas", "sign", "fred", "they", "lien", "kiev", "fops", "said", "lawn", "lind", "miff", "mass", "trig", "sins", "furl", "ruin", "sent", "cray", "maya", "clog", "puns", "silk", "axis", "grog", "jots", "dyer", "mope", "rand", "vend", "keen", "chou", "dose", "rain", "eats", "sped", "maui", "evan", "time", "todd", "skit", "lief", "sops", "outs", "moot", "faze", "biro", "gook", "fill", "oval", "skew", "veil", "born", "slob", "hyde", "twin", "eloy", "beat", "ergs", "sure", "kobe", "eggo", "hens", "jive", "flax", "mons", "dunk", "yest", "begs", "dial", "lodz", "burp", "pile", "much", "dock", "rene", "sago", "racy", "have", "yalu", "glow", "move", "peps", "hods", "kins", "salk", "hand", "cons", "dare", "myra", "sega", "type", "mari", "pelt", "hula", "gulf", "jugs", "flay", "fest", "spat", "toms", "zeno", "taps", "deny", "swag", "afro", "baud", "jabs", "smut", "egos", "lara", "toes", "song", "fray", "luis", "brut", "olen", "mere", "ruff", "slum", "glad", "buds", "silt", "rued", "gelt", "hive", "teem", "ides", "sink", "ands", "wisp", "omen", "lyre", "yuks", "curb", "loam", "darn", "liar", "pugs", "pane", "carl", "sang", "scar", "zeds", "claw", "berg", "hits", "mile", "lite", "khan", "erik", "slug", "loon", "dena", "ruse", "talk", "tusk", "gaol", "tads", "beds", "sock", "howe", "gave", "snob", "ahab", "part", "meir", "jell", "stir", "tels", "spit", "hash", "omit", "jinx", "lyra", "puck", "laue", "beep", "eros", "owed", "cede", "brew", "slue", "mitt", "jest", "lynx", "wads", "gena", "dank", "volt", "gray", "pony", "veld", "bask", "fens", "argo", "work", "taxi", "afar", "boon", "lube", "pass", "lazy", "mist", "blot", "mach", "poky", "rams", "sits", "rend", "dome", "pray", "duck", "hers", "lure", "keep", "gory", "chat", "runt", "jams", "lays", "posy", "bats", "hoff", "rock", "keri", "raul", "yves", "lama", "ramp", "vote", "jody", "pock", "gist", "sass", "iago", "coos", "rank", "lowe", "vows", "koch", "taco", "jinn", "juno", "rape", "band", "aces", "goal", "huck", "lila", "tuft", "swan", "blab", "leda", "gems", "hide", "tack", "porn", "scum", "frat", "plum", "duds", "shad", "arms", "pare", "chin", "gain", "knee", "foot", "line", "dove", "vera", "jays", "fund", "reno", "skid", "boys", "corn", "gwyn", "sash", "weld", "ruiz", "dior", "jess", "leaf", "pars", "cote", "zing", "scat", "nice", "dart", "only", "owls", "hike", "trey", "whys", "ding", "klan", "ross", "barb", "ants", "lean", "dopy", "hock", "tour", "grip", "aldo", "whim", "prom", "rear", "dins", "duff", "dell", "loch", "lava", "sung", "yank", "thar", "curl", "venn", "blow", "pomp", "heat", "trap", "dali", "nets", "seen", "gash", "twig", "dads", "emmy", "rhea", "navy", "haws", "mite", "bows", "alas", "ives", "play", "soon", "doll", "chum", "ajar", "foam", "call", "puke", "kris", "wily", "came", "ales", "reef", "raid", "diet", "prod", "prut", "loot", "soar", "coed", "celt", "seam", "dray", "lump", "jags", "nods", "sole", "kink", "peso", "howl", "cost", "tsar", "uric", "sore", "woes", "sewn", "sake", "cask", "caps", "burl", "tame", "bulk", "neva", "from", "meet", "webs", "spar", "fuck", "buoy", "wept", "west", "dual", "pica", "sold", "seed", "gads", "riff", "neck", "deed", "rudy", "drop", "vale", "flit", "romp", "peak", "jape", "jews", "fain", "dens", "hugo", "elba", "mink", "town", "clam", "feud", "fern", "dung", "newt", "mime", "deem", "inti", "gigs", "sosa", "lope", "lard", "cara", "smug", "lego", "flex", "doth", "paar", "moon", "wren", "tale", "kant", "eels", "muck", "toga", "zens", "lops", "duet", "coil", "gall", "teal", "glib", "muir", "ails", "boer", "them", "rake", "conn", "neat", "frog", "trip", "coma", "must", "mono", "lira", "craw", "sled", "wear", "toby", "reel", "hips", "nate", "pump", "mont", "died", "moss", "lair", "jibe", "oils", "pied", "hobs", "cads", "haze", "muse", "cogs", "figs", "cues", "roes", "whet", "boru", "cozy", "amos", "tans", "news", "hake", "cots", "boas", "tutu", "wavy", "pipe", "typo", "albs", "boom", "dyke", "wail", "woke", "ware", "rita", "fail", "slab", "owes", "jane", "rack", "hell", "lags", "mend", "mask", "hume", "wane", "acne", "team", "holy", "runs", "exes", "dole", "trim", "zola", "trek", "puma", "wacs", "veep", "yaps", "sums", "lush", "tubs", "most", "witt", "bong", "rule", "hear", "awry", "sots", "nils", "bash", "gasp", "inch", "pens", "fies", "juts", "pate", "vine", "zulu", "this", "bare", "veal", "josh", "reek", "ours", "cowl", "club", "farm", "teat", "coat", "dish", "fore", "weft", "exam", "vlad", "floe", "beak", "lane", "ella", "warp", "goth", "ming", "pits", "rent", "tito", "wish", "amps", "says", "hawk", "ways", "punk", "nark", "cagy", "east", "paul", "bose", "solo", "teed", "text", "hews", "snip", "lips", "emit", "orgy", "icon", "tuna", "soul", "kurd", "clod", "calk", "aunt", "bake", "copy", "acid", "duse", "kiln", "spec", "fans", "bani", "irma", "pads", "batu", "logo", "pack", "oder", "atop", "funk", "gide", "bede", "bibs", "taut", "guns", "dana", "puff", "lyme", "flat", "lake", "june", "sets", "gull", "hops", "earn", "clip", "fell", "kama", "seal", "diaz", "cite", "chew", "cuba", "bury", "yard", "bank", "byes", "apia", "cree", "nosh", "judo", "walk", "tape", "taro", "boot", "cods", "lade", "cong", "deft", "slim", "jeri", "rile", "park", "aeon", "fact", "slow", "goff", "cane", "earp", "tart", "does", "acts", "hope", "cant", "buts", "shin", "dude", "ergo", "mode", "gene", "lept", "chen", "beta", "eden", "pang", "saab", "fang", "whir", "cove", "perk", "fads", "rugs", "herb", "putt", "nous", "vane", "corm", "stay", "bids", "vela", "roof", "isms", "sics", "gone", "swum", "wiry", "cram", "rink", "pert", "heap", "sikh", "dais", "cell", "peel", "nuke", "buss", "rasp", "none", "slut", "bent", "dams", "serb", "dork", "bays", "kale", "cora", "wake", "welt", "rind", "trot", "sloe", "pity", "rout", "eves", "fats", "furs", "pogo", "beth", "hued", "edam", "iamb", "glee", "lute", "keel", "airy", "easy", "tire", "rube", "bogy", "sine", "chop", "rood", "elbe", "mike", "garb", "jill", "gaul", "chit", "dons", "bars", "ride", "beck", "toad", "make", "head", "suds", "pike", "snot", "swat", "peed", "same", "gaza", "lent", "gait", "gael", "elks", "hang", "nerf", "rosy", "shut", "glop", "pain", "dion", "deaf", "hero", "doer", "wost", "wage", "wash", "pats", "narc", "ions", "dice", "quay", "vied", "eons", "case", "pour", "urns", "reva", "rags", "aden", "bone", "rang", "aura", "iraq", "toot", "rome", "hals", "megs", "pond", "john", "yeps", "pawl", "warm", "bird", "tint", "jowl", "gibe", "come", "hold", "pail", "wipe", "bike", "rips", "eery", "kent", "hims", "inks", "fink", "mott", "ices", "macy", "serf", "keys", "tarp", "cops", "sods", "feet", "tear", "benz", "buys", "colo", "boil", "sews", "enos", "watt", "pull", "brag", "cork", "save", "mint", "feat", "jamb", "rubs", "roxy", "toys", "nosy", "yowl", "tamp", "lobs", "foul", "doom", "sown", "pigs", "hemp", "fame", "boor", "cube", "tops", "loco", "lads", "eyre", "alta", "aged", "flop", "pram", "lesa", "sawn", "plow", "aral", "load", "lied", "pled", "boob", "bert", "rows", "zits", "rick", "hint", "dido", "fist", "marc", "wuss", "node", "smog", "nora", "shim", "glut", "bale", "perl", "what", "tort", "meek", "brie", "bind", "cake", "psst", "dour", "jove", "tree", "chip", "stud", "thou", "mobs", "sows", "opts", "diva", "perm", "wise", "cuds", "sols", "alan", "mild", "pure", "gail", "wins", "offs", "nile", "yelp", "minn", "tors", "tran", "homy", "sadr", "erse", "nero", "scab", "finn", "mich", "turd", "then", "poem", "noun", "oxus", "brow", "door", "saws", "eben", "wart", "wand", "rosa", "left", "lina", "cabs", "rapt", "olin", "suet", "kalb", "mans", "dawn", "riel", "temp", "chug", "peal", "drew", "null", "hath", "many", "took", "fond", "gate", "sate", "leak", "zany", "vans", "mart", "hess", "home", "long", "dirk", "bile", "lace", "moog", "axes", "zone", "fork", "duct", "rico", "rife", "deep", "tiny", "hugh", "bilk", "waft", "swig", "pans", "with", "kern", "busy", "film", "lulu", "king", "lord", "veda", "tray", "legs", "soot", "ells", "wasp", "hunt", "earl", "ouch", "diem", "yell", "pegs", "blvd", "polk", "soda", "zorn", "liza", "slop", "week", "kill", "rusk", "eric", "sump", "haul", "rims", "crop", "blob", "face", "bins", "read", "care", "pele", "ritz", "beau", "golf", "drip", "dike", "stab", "jibs", "hove", "junk", "hoax", "tats", "fief", "quad", "peat", "ream", "hats", "root", "flak", "grit", "clap", "pugh", "bosh", "lock", "mute", "crow", "iced", "lisa", "bela", "fems", "oxes", "vies", "gybe", "huff", "bull", "cuss", "sunk", "pups", "fobs", "turf", "sect", "atom", "debt", "sane", "writ", "anon", "mayo", "aria", "seer", "thor", "brim", "gawk", "jack", "jazz", "menu", "yolk", "surf", "libs", "lets", "bans", "toil", "open", "aced", "poor", "mess", "wham", "fran", "gina", "dote", "love", "mood", "pale", "reps", "ines", "shot", "alar", "twit", "site", "dill", "yoga", "sear", "vamp", "abel", "lieu", "cuff", "orbs", "rose", "tank", "gape", "guam", "adar", "vole", "your", "dean", "dear", "hebe", "crab", "hump", "mole", "vase", "rode", "dash", "sera", "balk", "lela", "inca", "gaea", "bush", "loud", "pies", "aide", "blew", "mien", "side", "kerr", "ring", "tess", "prep", "rant", "lugs", "hobo", "joke", "odds", "yule", "aida", "true", "pone", "lode", "nona", "weep", "coda", "elmo", "skim", "wink", "bras", "pier", "bung", "pets", "tabs", "ryan", "jock", "body", "sofa", "joey", "zion", "mace", "kick", "vile", "leno", "bali", "fart", "that", "redo", "ills", "jogs", "pent", "drub", "slaw", "tide", "lena", "seep", "gyps", "wave", "amid", "fear", "ties", "flan", "wimp", "kali", "shun", "crap", "sage", "rune", "logs", "cain", "digs", "abut", "obit", "paps", "rids", "fair", "hack", "huns", "road", "caws", "curt", "jute", "fisk", "fowl", "duty", "holt", "miss", "rude", "vito", "baal", "ural", "mann", "mind", "belt", "clem", "last", "musk", "roam", "abed", "days", "bore", "fuze", "fall", "pict", "dump", "dies", "fiat", "vent", "pork", "eyed", "docs", "rive", "spas", "rope", "ariz", "tout", "game", "jump", "blur", "anti", "lisp", "turn", "sand", "food", "moos", "hoop", "saul", "arch", "fury", "rise", "diss", "hubs", "burs", "grid", "ilks", "suns", "flea", "soil", "lung", "want", "nola", "fins", "thud", "kidd", "juan", "heps", "nape", "rash", "burt", "bump", "tots", "brit", "mums", "bole", "shah", "tees", "skip", "limb", "umps", "ache", "arcs", "raft", "halo", "luce", "bahs", "leta", "conk", "duos", "siva", "went", "peek", "sulk", "reap", "free", "dubs", "lang", "toto", "hasp", "ball", "rats", "nair", "myst", "wang", "snug", "nash", "laos", "ante", "opal", "tina", "pore", "bite", "haas", "myth", "yugo", "foci", "dent", "bade", "pear", "mods", "auto", "shop", "etch", "lyly", "curs", "aron", "slew", "tyro", "sack", "wade", "clio", "gyro", "butt", "icky", "char", "itch", "halt", "gals", "yang", "tend", "pact", "bees", "suit", "puny", "hows", "nina", "brno", "oops", "lick", "sons", "kilo", "bust", "nome", "mona", "dull", "join", "hour", "papa", "stag", "bern", "wove", "lull", "slip", "laze", "roil", "alto", "bath", "buck", "alma", "anus", "evil", "dumb", "oreo", "rare", "near", "cure", "isis", "hill", "kyle", "pace", "comb", "nits", "flip", "clop", "mort", "thea", "wall", "kiel", "judd", "coop", "dave", "very", "amie", "blah", "flub", "talc", "bold", "fogy", "idea", "prof", "horn", "shoo", "aped", "pins", "helm", "wees", "beer", "womb", "clue", "alba", "aloe", "fine", "bard", "limo", "shaw", "pint", "swim", "dust", "indy", "hale", "cats", "troy", "wens", "luke", "vern", "deli", "both", "brig", "daub", "sara", "sued", "bier", "noel", "olga", "dupe", "look", "pisa", "knox", "murk", "dame", "matt", "gold", "jame", "toge", "luck", "peck", "tass", "calf", "pill", "wore", "wadi", "thur", "parr", "maul", "tzar", "ones", "lees", "dark", "fake", "bast", "zoom", "here", "moro", "wine", "bums", "cows", "jean", "palm", "fume", "plop", "help", "tuba", "leap", "cans", "back", "avid", "lice", "lust", "polo", "dory", "stew", "kate", "rama", "coke", "bled", "mugs", "ajax", "arts", "drug", "pena", "cody", "hole", "sean", "deck", "guts", "kong", "bate", "pitt", "como", "lyle", "siam", "rook", "baby", "jigs", "bret", "bark", "lori", "reba", "sups", "made", "buzz", "gnaw", "alps", "clay", "post", "viol", "dina", "card", "lana", "doff", "yups", "tons", "live", "kids", "pair", "yawl", "name", "oven", "sirs", "gyms", "prig", "down", "leos", "noon", "nibs", "cook", "safe", "cobb", "raja", "awes", "sari", "nerd", "fold", "lots", "pete", "deal", "bias", "zeal", "girl", "rage", "cool", "gout", "whey", "soak", "thaw", "bear", "wing", "nagy", "well", "oink", "sven", "kurt", "etna", "held", "wood", "high", "feta", "twee", "ford", "cave", "knot", "tory", "ibis", "yaks", "vets", "foxy", "sank", "cone", "pius", "tall", "seem", "wool", "flap", "gird", "lore", "coot", "mewl", "sere", "real", "puts", "sell", "nuts", "foil", "lilt", "saga", "heft", "dyed", "goat", "spew", "daze", "frye", "adds", "glen", "tojo", "pixy", "gobi", "stop", "tile", "hiss", "shed", "hahn", "baku", "ahas", "sill", "swap", "also", "carr", "manx", "lime", "debs", "moat", "eked", "bola", "pods", "coon", "lacy", "tube", "minx", "buff", "pres", "clew", "gaff", "flee", "burn", "whom", "cola", "fret", "purl", "wick", "wigs", "donn", "guys", "toni", "oxen", "wite", "vial", "spam", "huts", "vats", "lima", "core", "eula", "thad", "peon", "erie", "oats", "boyd", "cued", "olaf", "tams", "secs", "urey", "wile", "penn", "bred", "rill", "vary", "sues", "mail", "feds", "aves", "code", "beam", "reed", "neil", "hark", "pols", "gris", "gods", "mesa", "test", "coup", "heed", "dora", "hied", "tune", "doze", "pews", "oaks", "bloc", "tips", "maid", "goof", "four", "woof", "silo", "bray", "zest", "kiss", "yong", "file", "hilt", "iris", "tuns", "lily", "ears", "pant", "jury", "taft", "data", "gild", "pick", "kook", "colt", "bohr", "anal", "asps", "babe", "bach", "mash", "biko", "bowl", "huey", "jilt", "goes", "guff", "bend", "nike", "tami", "gosh", "tike", "gees", "urge", "path", "bony", "jude", "lynn", "lois", "teas", "dunn", "elul", "bonn", "moms", "bugs", "slay", "yeah", "loan", "hulk", "lows", "damn", "nell", "jung", "avis", "mane", "waco", "loin", "knob", "tyke", "anna", "hire", "luau", "tidy", "nuns", "pots", "quid", "exec", "hans", "hera", "hush", "shag", "scot", "moan", "wald", "ursa", "lorn", "hunk", "loft", "yore", "alum", "mows", "slog", "emma", "spud", "rice", "worn", "erma", "need", "bags", "lark", "kirk", "pooh", "dyes", "area", "dime", "luvs", "foch", "refs", "cast", "alit", "tugs", "even", "role", "toed", "caph", "nigh", "sony", "bide", "robs", "folk", "daft", "past", "blue", "flaw", "sana", "fits", "barr", "riot", "dots", "lamp", "cock", "fibs", "harp", "tent", "hate", "mali", "togs", "gear", "tues", "bass", "pros", "numb", "emus", "hare", "fate", "wife", "mean", "pink", "dune", "ares", "dine", "oily", "tony", "czar", "spay", "push", "glum", "till", "moth", "glue", "dive", "scad", "pops", "woks", "andy", "leah", "cusp", "hair", "alex", "vibe", "bulb", "boll", "firm", "joys", "tara", "cole", "levy", "owen", "chow", "rump", "jail", "lapp", "beet", "slap", "kith", "more", "maps", "bond", "hick", "opus", "rust", "wist", "shat", "phil", "snow", "lott", "lora", "cary", "mote", "rift", "oust", "klee", "goad", "pith", "heep", "lupe", "ivan", "mimi", "bald", "fuse", "cuts", "lens", "leer", "eyry", "know", "razz", "tare", "pals", "geek", "greg", "teen", "clef", "wags", "weal", "each", "haft", "nova", "waif", "rate", "katy", "yale", "dale", "leas", "axum", "quiz", "pawn", "fend", "capt", "laws", "city", "chad", "coal", "nail", "zaps", "sort", "loci", "less", "spur", "note", "foes", "fags", "gulp", "snap", "bogs", "wrap", "dane", "melt", "ease", "felt", "shea", "calm", "star", "swam", "aery", "year", "plan", "odin", "curd", "mira", "mops", "shit", "davy", "apes", "inky", "hues", "lome", "bits", "vila", "show", "best", "mice", "gins", "next", "roan", "ymir", "mars", "oman", "wild", "heal", "plus", "erin", "rave", "robe", "fast", "hutu", "aver", "jodi", "alms", "yams", "zero", "revs", "wean", "chic", "self", "jeep", "jobs", "waxy", "duel", "seek", "spot", "raps", "pimp", "adan", "slam", "tool", "morn", "futz", "ewes", "errs", "knit", "rung", "kans", "muff", "huhs", "tows", "lest", "meal", "azov", "gnus", "agar", "sips", "sway", "otis", "tone", "tate", "epic", "trio", "tics", "fade", "lear", "owns", "robt", "weds", "five", "lyon", "terr", "arno", "mama", "grey", "disk", "sept", "sire", "bart", "saps", "whoa", "turk", "stow", "pyle", "joni", "zinc", "negs", "task", "leif", "ribs", "malt", "nine", "bunt", "grin", "dona", "nope", "hams", "some", "molt", "smit", "sacs", "joan", "slav", "lady", "base", "heck", "list", "take", "herd", "will", "nubs", "burg", "hugs", "peru", "coif", "zoos", "nick", "idol", "levi", "grub", "roth", "adam", "elma", "tags", "tote", "yaws", "cali", "mete", "lula", "cubs", "prim", "luna", "jolt", "span", "pita", "dodo", "puss", "deer", "term", "dolt", "goon", "gary", "yarn", "aims", "just", "rena", "tine", "cyst", "meld", "loki", "wong", "were", "hung", "maze", "arid", "cars", "wolf", "marx", "faye", "eave", "raga", "flow", "neal", "lone", "anne", "cage", "tied", "tilt", "soto", "opel", "date", "buns", "dorm", "kane", "akin", "ewer", "drab", "thai", "jeer", "grad", "berm", "rods", "saki", "grus", "vast", "late", "lint", "mule", "risk", "labs", "snit", "gala", "find", "spin", "ired", "slot", "oafs", "lies", "mews", "wino", "milk", "bout", "onus", "tram", "jaws", "peas", "cleo", "seat", "gums", "cold", "vang", "dewy", "hood", "rush", "mack", "yuan", "odes", "boos", "jami", "mare", "plot", "swab", "borg", "hays", "form", "mesh", "mani", "fife", "good", "gram", "lion", "myna", "moor", "skin", "posh", "burr", "rime", "done", "ruts", "pays", "stem", "ting", "arty", "slag", "iron", "ayes", "stub", "oral", "gets", "chid", "yens", "snub", "ages", "wide", "bail", "verb", "lamb", "bomb", "army", "yoke", "gels", "tits", "bork", "mils", "nary", "barn", "hype", "odom", "avon", "hewn", "rios", "cams", "tact", "boss", "oleo", "duke", "eris", "gwen", "elms", "deon", "sims", "quit", "nest", "font", "dues", "yeas", "zeta", "bevy", "gent", "torn", "cups", "worm", "baum", "axon", "purr", "vise", "grew", "govs", "meat", "chef", "rest", "lame" };
	
	timer.reset();
	s.findLadders("sand", "acne", us);
	int t = timer.getus();
	cout << t << endl;

	unordered_set<string> dict;
	dict.insert("cat");
	dict.insert("cats");
	dict.insert("and");
	dict.insert("sand");
	dict.insert("dog");
	
	ListNode ln0(3), ln1(5), ln2(8);
	ln0.next = &ln1;
	ln1.next = &ln2;
	

	TreeNode tr(0);
	//s.isValidBST(&tr);

	return 0;
}
