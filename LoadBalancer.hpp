/**
 *  TODO:   * Threads could block each other: When a job thread starts another job-thread,
 *              the child one wont be executed, if the max number of threads is already reached.
 *              The child will wait for ever in the queue, the parent will wait for the child.
 *              Right now the constructor parameter <inception> needs to be set to true, if 
 *				the LoadBalancer is created within an already running job. If so, the maximum
 *				number of threads is altered by one (since the parent one should just be waiting
 *				for the jobs). There sure are better ways to solve this ... 
 *					* double list of running jobs, one static to check if this_thread is in there?
 *					* doing everythin global (static or singleton) and remove the parent one from
 *						the job list didn't work. Race conditions lead to the parent returning from
 *						start_jobs without the children being finished
 *			* What does it mean to "bind" a function, that happens with memory, scope, cache ...
 *			* Is it bad if a bound function is copied, destroyed, referenced, shared_ptr...ed or
 *				passed to another bound function?
 *				
**/
#pragma once
//#define DEBUG_LB

#ifdef DEBUG_LB
#define DEBUG_DO(x) x
#else
#define DEBUG_DO(x) do { } while (false)
#endif


#include <chrono>
#include <functional>
#include <memory>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

using namespace std;

typedef shared_ptr<thread> thread_ptr;



class LoadBalancer {
public:
	LoadBalancer(bool inception = false);
	~LoadBalancer() {};
	LoadBalancer(LoadBalancer const &);
	void operator=(LoadBalancer const &);
	void add_job(function<void(void)> func);
	void start_jobs();


private:
	void job_finished(thread::id id);
	void do_job(function<void(void)> func);


	chrono::milliseconds sleep_time;
	queue<thread::id> finished_jobs;
	queue<function<void(void)> > todo_list;
	unordered_map<thread::id, thread_ptr> running_threads;
	mutex local_mutex;
	bool inception;

	static int running_thread_count;
	static int max_threads;
	static mutex global_mutex;
};